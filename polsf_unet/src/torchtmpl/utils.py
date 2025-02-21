# coding: utf-8
# MIT License

# Copyright (c) 2023 Jeremy Fix

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Standard imports
import os
from typing import Tuple
import inspect
import warnings

# External imports
import torch
import torch.nn as nn
import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    jaccard_score,
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
)
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score


def compute_metrics(
    predictions, ground_truth, ignore_index, num_classes=None, test=False
):
    # Create a mask to filter out the ignore_index
    mask = ground_truth != ignore_index
    filtered_predictions = predictions[mask]
    filtered_ground_truth = ground_truth[mask]

    # Calculate Jaccard Index (IoU)
    iou = jaccard_score(filtered_ground_truth, filtered_predictions, average="weighted")

    # Calculate overall accuracy
    accuracy = accuracy_score(filtered_ground_truth, filtered_predictions)

    if test:
        conf_matrix = confusion_matrix(
            filtered_ground_truth,
            filtered_predictions,
            labels=np.setdiff1d(np.arange(0, num_classes), np.array([ignore_index])),
            normalize="true",
        )
        # Calculate average accuracy (balanced accuracy)
        average_accuracy = balanced_accuracy_score(
            filtered_ground_truth, filtered_predictions
        )

        # Calculate Cohen's kappa
        kappa = cohen_kappa_score(filtered_ground_truth, filtered_predictions)

        return iou, accuracy, average_accuracy, kappa, conf_matrix
    else:
        return iou, accuracy


def normalize_confusion_matrix(conf_matrix):
    row_sums = conf_matrix.sum(axis=1, keepdims=True)  # Sum of each row
    normalized_matrix = conf_matrix / (row_sums + 1e-6)  # Avoid division by zero
    return normalized_matrix


def compute_batch_iou(predictions, labels, ignore_index=None):
    mask = labels != ignore_index
    filtered_predictions = predictions[mask]
    filtered_ground_truth = labels[mask]

    # Calculate Jaccard Index (IoU)
    return jaccard_score(
        filtered_ground_truth, filtered_predictions, average="weighted"
    )


# Function to compute confusion matrix for a batch
def compute_batch_confusion_matrix(predictions, labels, size, ignore_index):
    mask = labels != ignore_index
    predictions = predictions[mask]
    labels = labels[mask]
    return confusion_matrix(
        labels,
        predictions,
        labels=size,
    )


# Function to compute overall accuracy
def compute_overall_accuracy(conf_matrix):
    return np.trace(conf_matrix) / conf_matrix.sum()


# Function to compute Cohen's Kappa
def compute_kappa(conf_matrix):
    row_sums = conf_matrix.sum(axis=1)
    col_sums = conf_matrix.sum(axis=0)
    expected_agreement = (row_sums @ col_sums) / conf_matrix.sum() ** 2
    observed_agreement = np.trace(conf_matrix) / conf_matrix.sum()
    return (observed_agreement - expected_agreement) / (1 - expected_agreement)


def compute_iou(confusion_matrix):

    n_classes = confusion_matrix.shape[0]
    iou_per_class = []

    for c in range(n_classes):
        # True Positive for class c
        tp = confusion_matrix[c, c]
        # False Positive: sum of predicted as class c but not true class c
        fp = confusion_matrix[:, c].sum() - tp
        # False Negative: sum of true class c but not predicted as class c
        fn = confusion_matrix[c, :].sum() - tp
        # Intersection = TP, Union = TP + FP + FN
        union = tp + fp + fn
        # Avoid division by zero
        iou = tp / union if union > 0 else 0.0
        iou_per_class.append(iou)

    # Mean IoU
    mean_iou = np.mean(iou_per_class)

    return iou_per_class, mean_iou


# Function to compute classification metrics
def compute_classification_metrics(conf_matrix, ignore_index=None):
    if ignore_index is not None:
        conf_matrix = np.delete(conf_matrix, ignore_index, axis=0)
        conf_matrix = np.delete(conf_matrix, ignore_index, axis=1)

    precision = np.diag(conf_matrix) / (conf_matrix.sum(axis=0) + 1e-6)
    recall = np.diag(conf_matrix) / (conf_matrix.sum(axis=1) + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return {
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "macro_precision": precision.mean(),
        "macro_recall": recall.mean(),
        "macro_f1": f1.mean(),
    }


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    f_loss: nn.Module,
    optim: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    device: torch.device,
    number_classes,
    epoch: int,
    ignore_index,
    max_norm=2.5,
) -> dict:
    """
    Run the training loop for nsteps minibatches of the dataloader

    Arguments:
        model: the model to train
        loader: an iterable dataloader
        f_loss (nn.Module): the loss
        optim : an optimizing algorithm
        device: the device on which to run the code

    Returns:
        A dictionary with averaged training metrics
    """
    model.train()

    loss_avg = 0
    gradient_norm = 0
    num_samples = 0
    num_batches = 0
    softmax = nn.Softmax(dim=1)

    size = np.setdiff1d(np.arange(0, number_classes), np.array([ignore_index]))
    conf_matrix_accum = np.zeros((len(size), len(size)))

    for data in tqdm.tqdm(loader):
        if isinstance(data, tuple) or isinstance(data, list):
            inputs, labels = data
            labels = labels.to(device)
        else:
            inputs = data

        inputs = Variable(inputs, requires_grad=False).to(device)
        # Forward propagate through the model
        pred_outputs = model(inputs)

        pred_outputs = softmax(torch.abs(pred_outputs).type(torch.float64))

        loss = f_loss(
            pred_outputs,
            labels.type(torch.int64),
        )

        predictions_flat = pred_outputs.argmax(dim=1).cpu().numpy().flatten()
        labels_flat = labels.cpu().numpy().flatten()

        # Update confusion matrix
        batch_cm = compute_batch_confusion_matrix(
            predictions=predictions_flat,
            labels=labels_flat,
            ignore_index=ignore_index,
            size=size,
        )
        conf_matrix_accum += batch_cm

        # Backward pass and update
        optim.zero_grad()
        loss.backward()

        # Clip gradients to prevent the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=max_norm, norm_type=2
        )

        # Compute the norm of the gradients
        total_norm = np.sqrt(
            sum(
                p.grad.data.norm(2).item() ** 2
                for p in model.parameters()
                if p.grad is not None
            )
        )
        gradient_norm += total_norm

        optim.step()
        if isinstance(
            scheduler,
            (
                torch.optim.lr_scheduler.CyclicLR,
                torch.optim.lr_scheduler.OneCycleLR,
                torch.optim.lr_scheduler.CosineAnnealingLR,
            ),
        ):
            scheduler.step()
        elif isinstance(
            scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        ):
            scheduler.step(epoch + num_batches / len(loader))

        num_samples += inputs.shape[0]
        num_batches += 1
        loss_avg += inputs.shape[0] * loss.item()

        del loss, pred_outputs, inputs

    torch.cuda.empty_cache()

    metrics = {
        "train_loss": loss_avg / num_samples,
        "gradient_norm": gradient_norm / num_batches,
    }

    overall_accuracy = compute_overall_accuracy(conf_matrix_accum)
    kappa_score = compute_kappa(conf_matrix_accum)
    metrics_classif = compute_classification_metrics(conf_matrix_accum, ignore_index)
    metrics["train_overall_accuracy"] = 100 * overall_accuracy
    metrics["train_kappa_score"] = 100 * kappa_score
    metrics["train_macro_precision"] = 100 * metrics_classif["macro_precision"]
    metrics["train_macro_recall"] = 100 * metrics_classif["macro_recall"]
    metrics["train_macro_f1"] = 100 * metrics_classif["macro_f1"]
    metrics["train_precision_per_class"] = 100 * metrics_classif["precision_per_class"]
    metrics["train_recall_per_class"] = 100 * metrics_classif["recall_per_class"]
    metrics["train_f1_per_class"] = 100 * metrics_classif["f1_per_class"]

    iou_classes, mean_iou = compute_iou(conf_matrix_accum)
    metrics["train_iou_per_class"] = 100 * iou_classes
    metrics["train_mean_iou"] = 100 * mean_iou

    return metrics


def valid_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    f_loss: nn.Module,
    device: torch.device,
    number_classes,
    ignore_index=-100,
) -> dict:
    """
    Run the valid loop for n_valid_batches minibatches of the dataloader

    Arguments:
        model: the model to evaluate
        loader: an iterable dataloader
        f_loss: the loss
        device: the device on which to run the code

    Returns:
        A dictionary with averaged valid metrics
    """
    model.eval()

    loss_avg = 0
    num_samples = 0
    num_batches = 0
    softmax = nn.Softmax(dim=1)

    size = np.setdiff1d(np.arange(0, number_classes), np.array([ignore_index]))
    conf_matrix_accum = np.zeros((len(size), len(size)))

    with torch.no_grad():
        for data in tqdm.tqdm(loader):
            if isinstance(data, tuple) or isinstance(data, list):
                inputs, labels = data
                labels = labels.to(device)
            else:
                inputs = data
            inputs = Variable(inputs).to(device)

            # Forward propagate through the model
            pred_outputs = model(inputs)

            pred_outputs = softmax(torch.abs(pred_outputs).type(torch.float64))

            loss = f_loss(
                pred_outputs,
                labels.type(torch.int64),
            )

            predictions_flat = pred_outputs.argmax(dim=1).cpu().numpy().flatten()
            labels_flat = labels.cpu().numpy().flatten()

            # Update confusion matrix
            batch_cm = compute_batch_confusion_matrix(
                predictions=predictions_flat,
                labels=labels_flat,
                ignore_index=ignore_index,
                size=size,
            )
            conf_matrix_accum += batch_cm

            num_samples += inputs.shape[0]
            num_batches += 1
            loss_avg += inputs.shape[0] * loss.item()

    metrics = {"valid_loss": loss_avg / num_samples}

    overall_accuracy = compute_overall_accuracy(conf_matrix_accum)
    kappa_score = compute_kappa(conf_matrix_accum)
    metrics_classif = compute_classification_metrics(conf_matrix_accum, ignore_index)
    metrics["valid_overall_accuracy"] = 100 * overall_accuracy
    metrics["valid_kappa_score"] = 100 * kappa_score
    metrics["valid_macro_precision"] = 100 * metrics_classif["macro_precision"]
    metrics["valid_macro_recall"] = 100 * metrics_classif["macro_recall"]
    metrics["valid_macro_f1"] = 100 * metrics_classif["macro_f1"]
    metrics["valid_precision_per_class"] = 100 * metrics_classif["precision_per_class"]
    metrics["valid_recall_per_class"] = 100 * metrics_classif["recall_per_class"]
    metrics["valid_f1_per_class"] = 100 * metrics_classif["f1_per_class"]

    # Additional segmentation-specific metrics
    iou_classes, mean_iou = compute_iou(conf_matrix_accum)
    metrics["valid_iou_per_class"] = 100 * iou_classes
    metrics["valid_mean_iou"] = 100 * mean_iou
    return metrics


def test_epoch(
    model,
    loader,
    device,
    number_classes,
    ignore_index,
    num_samples_to_visualize=5,
):
    model.eval()
    model.to(device)

    num_samples = 0
    num_batches = 0
    softmax = nn.Softmax(dim=1)

    size = np.setdiff1d(np.arange(0, number_classes), np.array([ignore_index]))
    conf_matrix_accum = np.zeros((len(size), len(size)))

    to_be_vizualized = []

    with torch.no_grad():
        for data in tqdm.tqdm(loader):
            if isinstance(data, tuple) or isinstance(data, list):
                inputs, labels = data
                labels = labels.to(device)
            else:
                inputs = data
            inputs = Variable(inputs).to(device)

            # Forward propagate through the model
            pred_outputs = model(inputs)

            pred_outputs = softmax(torch.abs(pred_outputs).type(torch.float64))

            if num_samples_to_visualize > 0:
                num_samples_to_visualize -= 1
                random_index = np.random.randint(0, inputs.shape[0])

                to_be_vizualized.append(
                    (
                        inputs[random_index].cpu().numpy(),
                        labels[random_index].cpu().numpy(),
                        pred_outputs.argmax(dim=1)[random_index].cpu().numpy(),
                    )
                )

            predictions_flat = pred_outputs.argmax(dim=1).cpu().numpy().flatten()
            labels_flat = labels.cpu().numpy().flatten()

            # Update confusion matrix
            batch_cm = compute_batch_confusion_matrix(
                predictions=predictions_flat,
                labels=labels_flat,
                ignore_index=ignore_index,
                size=size,
            )

            conf_matrix_accum += batch_cm

            num_samples += inputs.shape[0]
            num_batches += 1

    metrics = {}

    overall_accuracy = compute_overall_accuracy(conf_matrix_accum)
    kappa_score = compute_kappa(conf_matrix_accum)
    metrics_classif = compute_classification_metrics(conf_matrix_accum, ignore_index)
    conf_matrix_accum = normalize_confusion_matrix(conf_matrix_accum)

    metrics["test_overall_accuracy"] = 100 * overall_accuracy
    metrics["test_kappa_score"] = 100 * kappa_score
    metrics["test_macro_precision"] = 100 * metrics_classif["macro_precision"]
    metrics["test_macro_recall"] = 100 * metrics_classif["macro_recall"]
    metrics["test_macro_f1"] = 100 * metrics_classif["macro_f1"]
    metrics["test_precision_per_class"] = 100 * metrics_classif["precision_per_class"]
    metrics["test_recall_per_class"] = 100 * metrics_classif["recall_per_class"]
    metrics["test_f1_per_class"] = 100 * metrics_classif["f1_per_class"]
    iou_classes, mean_iou = compute_iou(conf_matrix_accum)
    metrics["test_iou_per_class"] = 100 * iou_classes
    metrics["test_mean_iou"] = 100 * mean_iou

    return metrics, to_be_vizualized, conf_matrix_accum


def one_forward(model, loader, device):
    outputs = []
    model.eval()
    model.to(device)

    softmax = nn.Softmax(dim=1)

    list_of_indices = []

    with torch.no_grad():
        for _, data in enumerate(tqdm.tqdm(loader)):

            # Handle different data structures (tuple, list, or otherwise)
            if isinstance(data, (tuple, list)):
                if len(data) == 2:
                    inputs, labels = data  # For standard datasets
                elif len(data) == 3:
                    inputs, labels, idx = data  # For wrapped datasets
                    list_of_indices.extend(idx.cpu().numpy().tolist())
                else:
                    raise ValueError("Unexpected data format in loader.")
            else:
                inputs = data
                labels = None
            # Need to adapt the wrapper and the collect of the indices for reconstruction datasets

            inputs = Variable(inputs).to(device)

            # Forward propagate through the model
            pred_outputs = model(inputs)

            pred_outputs = (
                softmax(torch.abs(pred_outputs).type(torch.float64))
                .argmax(dim=1)
                .cpu()
                .numpy()
            )
            outputs.extend(pred_outputs)

    return (
        outputs,
        list_of_indices,
    )


class ModelCheckpoint(object):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        savepath: str,
        num_input_dims: int,
        min_is_best: bool = True,
    ) -> None:
        """
        Early stopping callback

        Arguments:
            model: the model to save
            savepath: the location where to save the model's parameters
            num_input_dims: the number of dimensions for the input tensor (required for onnx export)
            min_is_best: whether the min metric or the max metric as the best
        """
        self.model = model
        self.optimizer = optimizer
        self.savepath = savepath
        self.num_input_dims = num_input_dims
        self.best_score = None
        if min_is_best:
            self.is_better = self.lower_is_better
        else:
            self.is_better = self.higher_is_better

    def lower_is_better(self, score: float) -> bool:
        """
        Test if the provided score is lower than the best score found so far

        Arguments:
            score: the score to test

        Returns:
            res : is the provided score lower than the best score so far ?
        """
        return self.best_score is None or score < self.best_score

    def higher_is_better(self, score: float) -> bool:
        """
        Test if the provided score is higher than the best score found so far

        Arguments:
            score: the score to test

        Returns:
            res : is the provided score higher than the best score so far ?
        """
        return self.best_score is None or score > self.best_score

    def update(self, score: float, epoch: int) -> bool:
        """
        If the provided score is better than the best score registered so far,
        saves the model's parameters on disk as a pytorch tensor

        Arguments:
            score: the new score to consider

        Returns:
            res: whether or not the provided score is better than the best score
                 registered so far
        """
        if self.is_better(score):
            self.model.eval()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": score,
                },
                os.path.join(self.savepath, "best_model.pt"),
            )

            self.best_score = score
            return True
        return False


def generate_unique_logpath(logdir: str, raw_run_name: str) -> str:
    """
    Generate a unique directory name based on the highest existing suffix in directory names
    and create it if necessary.

    Arguments:
        logdir: the prefix directory
        raw_run_name: the base name

    Returns:
        log_path: a non-existent path like logdir/raw_run_name_x
                  where x is an int that is higher than any existing suffix.
    """
    highest_num = -1
    for item in os.listdir(logdir):
        if item.startswith(raw_run_name + "_") and os.path.isdir(
            os.path.join(logdir, item)
        ):
            try:
                suffix = int(item.split("_")[-1])
                highest_num = max(highest_num, suffix)
            except ValueError:
                # If conversion to int fails, ignore the directory name
                continue

    # The new directory name should be one more than the highest found
    new_num = highest_num + 1
    run_name = f"{raw_run_name}_{new_num}"
    log_path = os.path.join(logdir, run_name)
    os.makedirs(log_path)

    return log_path
