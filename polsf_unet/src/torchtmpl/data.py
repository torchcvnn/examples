import numpy as np
import logging
import random
import pathlib
from os import path, listdir
import shutil

import torch
import torch.nn as nn
import torchcvnn as cvnn
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from torchcvnn.datasets import PolSFDataset
from torchcvnn.transforms import FFTResize, PolSARtoTensor, ToTensor, Unsqueeze
import torchvision

# Constants for ignore index
IGNORE_INDEX = -100


class GenericDatasetWrapper(Dataset):
    def __init__(self, dataset):
        """
        A generic dataset wrapper that works with any dataset class.

        Args:
            dataset: An instance of a dataset class (e.g., CIFAR10, MNIST, etc.).
        """
        self.dataset = dataset

    def __getitem__(self, index):
        """
        Fetch an item from the dataset.

        Args:
            index: Index of the item to fetch.

        Returns:
            A tuple containing (data, target, index).
        """
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            Length of the dataset.
        """
        return len(self.dataset)


def get_transform_instance(transform_name, name_dataset, size):
    # Split the transform_name string on commas and strip whitespace
    transform_names = [name.strip() for name in transform_name.split(",")]

    # Get the classes from the module based on the transform names
    transform_instances = []

    transform_instances.append(PolSARtoTensor())

    transform_instances.append(ToTensor())

    for name in transform_names:
        try:
            TransformClass = getattr(cvnn.transforms, name)
        except AttributeError:
            TransformClass = getattr(torchvision.transforms, name)
        transform_instances.append(TransformClass())

    # If there's more than one transform, compose them
    if len(transform_instances) > 1:
        return torchvision.transforms.Compose(transform_instances)
    else:
        return transform_instances[0]


def calculate_class_distribution(masks: list, num_classes: int) -> np.ndarray:
    distributions = [
        np.histogram(mask, bins=np.arange(num_classes + 1))[0] / mask.size
        for mask in masks
    ]
    return np.array(distributions)


def stratify_masks(
    masks: list, num_classes: int, n_clusters: int = 10, random_state: int = 0
) -> np.ndarray:
    distributions = calculate_class_distribution(masks, num_classes)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(distributions)
    return kmeans.labels_


def get_dataloaders(data_config: dict, use_cuda: bool) -> tuple:
    (
        img_size,
        img_stride,
        valid_ratio,
        test_ratio,
        batch_size,
        num_workers,
        name_dataset,
        trainpath,
        transform,
    ) = extract_data_config(data_config)
    assert valid_ratio + test_ratio < 1.0

    class_weights, num_channels, num_classes, ignore_index = (
        None,
        None,
        None,
        IGNORE_INDEX,
    )

    ignore_index = 0  # Set ignore index to 0

    logging.info("  - Dataset creation")

    input_transform = get_transform_instance(transform, name_dataset, img_size)

    train_dataset, valid_dataset, test_dataset, _, _, _, _ = prepare_polsfdataset(
        trainpath,
        img_size,
        img_stride,
        input_transform,
        valid_ratio,
        test_ratio,
        data_config,
    )

    num_classes = len(train_dataset.dataset.classes)

    class_weights = compute_class_weights(train_dataset, num_classes, ignore_index)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    num_channels = get_num_channels(train_loader)
    img_size = train_loader.dataset[0][0].shape[-1]

    return (
        train_loader,
        valid_loader,
        test_loader,
        class_weights,
        num_classes,
        num_channels,
        img_size,
        ignore_index,
    )


def get_full_image_dataloader(data_config: dict) -> tuple:
    (
        img_size,
        _,
        valid_ratio,
        test_ratio,
        batch_size,
        num_workers,
        name_dataset,
        trainpath,
        transform,
    ) = extract_data_config(data_config)

    nsamples_per_cols, nsamples_per_rows = None, None

    logging.info("  - Dataset creation")

    input_transform = get_transform_instance(transform, name_dataset, img_size)

    (
        train_dataset,
        valid_dataset,
        test_dataset,
        base_dataset,
        train_indices,
        valid_indices,
        test_indices,
    ) = prepare_polsfdataset(
        trainpath,
        img_size,
        img_size,
        input_transform,
        valid_ratio,
        test_ratio,
        data_config,
    )
    nsamples_per_cols = base_dataset.alos_dataset.nsamples_per_cols
    nsamples_per_rows = base_dataset.alos_dataset.nsamples_per_rows

    wrapped_dataset = GenericDatasetWrapper(base_dataset)

    data_loader = DataLoader(
        wrapped_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    indices = [train_indices, valid_indices, test_indices]

    return (
        data_loader,
        nsamples_per_cols,
        nsamples_per_rows,
        indices,
    )


def extract_data_config(data_config: dict) -> tuple:
    """
    Extracts necessary fields from the data configuration dictionary.
    """
    if "img_size" not in data_config.keys():
        img_size = None
    else:
        img_size = (data_config["img_size"], data_config["img_size"])

    if "img_stride" not in data_config.keys():
        img_stride = None
    else:
        img_stride = (data_config["img_stride"], data_config["img_stride"])

    valid_ratio = data_config["valid_ratio"]
    test_ratio = data_config["test_ratio"]
    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]
    name_dataset = data_config["dataset"]["name"]
    transform = data_config["transform"]
    trainpath = path.expandvars(
        data_config["dataset"]["trainpath"]
    )  # apply the substitution if an environment variable is used
    return (
        img_size,
        img_stride,
        valid_ratio,
        test_ratio,
        batch_size,
        num_workers,
        name_dataset,
        trainpath,
        transform,
    )


def prepare_polsfdataset(
    trainpath,
    img_size,
    img_stride,
    input_transform,
    valid_ratio,
    test_ratio,
    data_config,
):
    base_dataset = eval(
        f"{data_config['dataset']['name']}(root=trainpath, transform=input_transform, patch_size=img_size, patch_stride=img_stride)"
    )
    num_classes = len(base_dataset.classes)
    masks = [base_dataset[i][1] for i in range(len(base_dataset))]

    strat_labels = stratify_masks(masks, num_classes)
    train_indices, temp_indices = train_test_split(
        list(range(len(base_dataset))),
        stratify=strat_labels,
        test_size=(valid_ratio + test_ratio),
    )

    temp_masks = [strat_labels[i] for i in temp_indices]
    valid_indices, test_indices = train_test_split(
        temp_indices,
        stratify=temp_masks,
        test_size=(test_ratio / (valid_ratio + test_ratio)),
    )

    train_dataset = Subset(base_dataset, train_indices)
    valid_dataset = Subset(base_dataset, valid_indices)
    test_dataset = Subset(base_dataset, test_indices)

    logging.info(f"  - Training set: {len(train_dataset)} samples")
    logging.info(f"  - Validation set: {len(valid_dataset)} samples")
    logging.info(f"  - Test set: {len(test_dataset)} samples")

    return (
        train_dataset,
        valid_dataset,
        test_dataset,
        base_dataset,
        train_indices,
        valid_indices,
        test_indices,
    )


def compute_class_weights(train_dataset, num_classes, ignore_index):
    if isinstance(train_dataset[0][1], int):
        all_labels = torch.tensor(
            [train_dataset[idx][1] for idx in range(len(train_dataset))]
        )
    elif isinstance(train_dataset[0][1], (torch.Tensor, np.ndarray)):
        all_labels = torch.cat(
            [
                torch.from_numpy(train_dataset[idx][1].flatten())
                for idx in range(len(train_dataset))
            ]
        )
    class_counts = torch.bincount(all_labels)

    if ignore_index > 0:
        # Exclude the count of the unlabeled class if necessary (assuming class 0 is unlabeled)
        class_counts = class_counts[1:] if num_classes > 1 else class_counts

    total_count = class_counts.sum()

    # Compute class weights
    class_weights = (
        (total_count / (num_classes - (1 if ignore_index > 0 else 0)) / class_counts)
        if num_classes > 1
        else np.array([1.0])
    )

    if ignore_index > 0:
        class_weights = np.concatenate(
            (np.array([0.0]), class_weights), dtype=np.float32
        )
        class_weights = torch.from_numpy(class_weights).type(torch.float32)

    return class_weights


def get_num_channels(loader):
    """
    Get the number of channels from the first image in the dataset.
    """
    return loader.dataset[0][0].shape[0]


def reassemble_image(
    segments,
    samples_per_col,
    samples_per_row,
    num_channels,
    segment_size,
    real_indices,
    sets_indices=None,
):
    """
    Reassemble an image from its segments using real_indices to determine their positions.

    Args:
        segments: List or array of image segments.
        samples_per_col: Number of segments per column in the reassembled image.
        samples_per_row: Number of segments per row in the reassembled image.
        num_channels: Number of channels in the image.
        segment_size: Height/width of each square segment.
        real_indices: List of real indices corresponding to the segments.
        sets_indices: List of sets of indices for mask assignment (optional).

    Returns:
        reassembled_image: The reconstructed image tensor.
        mask: A mask indicating the set each segment belongs to (if sets_indices is provided).
    """
    # Calculate total image dimensions
    img_height = samples_per_row * segment_size
    img_width = samples_per_col * segment_size

    # Initialize the empty image tensor with the correct shape
    reassembled_image = np.zeros(
        (num_channels, img_height, img_width), dtype=segments[0].dtype
    )
    if sets_indices is None:
        mask = None
    else:
        mask = np.zeros_like(reassembled_image, dtype=np.uint8)

    # Map real_indices to their positions
    index_to_position = {
        real_index: (row, col)
        for row in range(samples_per_row)
        for col in range(samples_per_col)
        for real_index in [row * samples_per_col + col]
    }

    # Place each segment into the correct position
    for segment_index, real_index in enumerate(real_indices):
        if real_index not in index_to_position:
            raise ValueError(
                f"Real index {real_index} is out of bounds for the image grid."
            )

        # Get the target row and column
        row, col = index_to_position[real_index]
        h_start = row * segment_size
        w_start = col * segment_size

        # Insert the segment into the image
        reassembled_image[
            :, h_start : h_start + segment_size, w_start : w_start + segment_size
        ] = segments[segment_index]

        # Update the mask if sets_indices is provided
        if mask is not None:
            if real_index in sets_indices[0]:
                mask[
                    :,
                    h_start : h_start + segment_size,
                    w_start : w_start + segment_size,
                ] = 0
            elif real_index in sets_indices[1]:
                mask[
                    :,
                    h_start : h_start + segment_size,
                    w_start : w_start + segment_size,
                ] = 1
            elif real_index in sets_indices[2]:
                mask[
                    :,
                    h_start : h_start + segment_size,
                    w_start : w_start + segment_size,
                ] = 2

    return reassembled_image, mask
