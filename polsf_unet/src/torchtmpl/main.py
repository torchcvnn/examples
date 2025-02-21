# coding: utf-8

# Standard imports
import logging
import sys
import pathlib
import random
from os import path, makedirs
import copy
from typing import Union

# External imports
import yaml
import wandb
import torch
import numpy as np
import math
import tqdm
from PIL import Image
import torch.nn as nn
import torchinfo
import torchcvnn.nn.modules as c_nn

# Local imports
from . import data as dt
from . import models
from . import optim
from . import utils
from . import visualization as vis
import torchtmpl as tl
from torchcvnn.datasets import PolSFDataset


def init_weights(m: nn.Module) -> None:
    """
    Initialize weights for the given module.
    """
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        c_nn.init.complex_kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def seed_everything(seed: int) -> None:
    """
    Seed all random number generators to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(
    checkpoint: torch.tensor,
    config: dict,
    num_classes: int,
    num_channels: int,
    img_size: int,
) -> nn.Module:
    """
    Load a pretrained model from the given path.
    """

    model = models.build_model(
        config,
        num_classes=num_classes,
        num_channels=num_channels,
        img_size=img_size,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def init_model(
    config: dict,
    num_classes: int,
    num_channels: int,
    img_size: int,
) -> nn.Module:
    """
    Initialize a model based on the given configuration.
    """
    model = models.build_model(
        config,
        num_classes=num_classes,
        num_channels=num_channels,
        img_size=img_size,
    )
    model.apply(init_weights)
    return model


def configure_wandb_logging(config: dict) -> None:
    """
    Configure and initialize Weights & Biases (wandb) logging.
    """
    if "wandb" in config["logging"]:
        wandb_config = config["logging"]["wandb"]
        project_name = wandb_config["project"]
        entity_name = wandb_config["entity"]
        run_id = wandb_config.get("run_id")

        if config["pretrained"]:
            wandb.init(
                project=project_name, entity=entity_name, resume="must", id=run_id
            )
            wandb_log = wandb.log
        else:
            config_cp = copy.deepcopy(config)
            config_cp = remove_wandb_tags(config_cp)
            tags = generate_tags(config_cp)

            # Ensure tags are limited to 64 characters
            tags = [tag if len(tag) <= 64 else tag[:61] + "..." for tag in tags]

            wandb.init(
                project=project_name, entity=entity_name, config=config_cp, tags=tags
            )
            config_cp["logging"]["wandb"]["run_id"] = wandb.run.id
            config["logging"]["wandb"]["run_id"] = wandb.run.id
            wandb_log = wandb.log
            wandb_log(config_cp)
        logging.info(f"Will be recording in wandb run name: {wandb.run.name}")
    else:
        wandb_log = None

    return wandb_log


def flatten_config(config, parent_key="", sep="."):
    """
    Recursively flattens a nested dictionary by concatenating keys.

    Args:
        config (dict): The configuration dictionary to flatten.
        parent_key (str): The base key to use for concatenation (used in recursion).
        sep (str): The separator to use between concatenated keys.

    Returns:
        dict: A flattened dictionary where nested keys are concatenated.
    """
    items = []

    for key, value in config.items():
        # Handle keys containing the separator character
        if sep in key:
            key = key.replace(sep, f"_{sep}_")

        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        # Recursively flatten dictionaries
        if isinstance(value, dict):
            if value:  # Skip empty dictionaries
                items.extend(flatten_config(value, new_key, sep=sep).items())
        elif isinstance(value, (list, tuple)):
            # Join list/tuple elements into a string for better representation
            items.append((new_key, ", ".join(map(str, value))))
        elif value is None:
            # Handle None values
            items.append((new_key, "None"))
        else:
            # Convert other types of values to strings
            items.append((new_key, str(value)))

    return dict(items)


def generate_tags(config):
    """
    Generates a list of formatted tags from a dynamic config dictionary.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        list: A list of formatted tags in the form of "key: value".
    """
    flat_config = flatten_config(config)
    tags = [f"{key}: {value}" for key, value in flat_config.items()]
    return tags


def remove_wandb_tags(config) -> None:

    config["logging"].pop("logdir", None)
    config.pop("seed", None)
    config.pop("pretrained", None)
    config.pop("world_size", None)

    return config


def load_config(config_path: str, command: str) -> dict:
    """
    Load configuration from a YAML file.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    if command != "train":
        path_to_run = sys.argv[3]
        config.update(yaml.safe_load(open(path_to_run + "/config.yml", "r")))
        config["pretrained"] = True

    return config


def load(config: dict) -> tuple:
    """
    Load model, optimizer, loss function, dataloaders, and other components based on the configuration.
    """
    log_path = config["logging"]["logdir"]

    seed = (
        config["seed"] if config["pretrained"] else math.floor(random.random() * 10000)
    )
    config["seed"] = seed
    seed_everything(seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    wandb_log = configure_wandb_logging(config)

    # Load the checkpoint if needed
    if config["pretrained"]:
        checkpoint_path = (
            log_path + "/last_model.pt"
        )  # to load the last model checkpoint to continue training
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)

    # Build the dataloaders
    logging.info("= Building the dataloaders")
    data_config = config["data"]
    (
        train_loader,
        valid_loader,
        _,
        class_weights,
        num_classes,
        num_channels,
        img_size,
        ignore_index,
    ) = dt.get_dataloaders(data_config, use_cuda)

    if len(next(iter(train_loader))) == 2:
        input_size = next(iter(train_loader))[0].shape
    else:
        input_size = next(iter(train_loader)).shape

    # Build the model
    logging.info("= Model")

    if config["pretrained"]:
        model = load_model(
            checkpoint,
            config,
            num_classes=num_classes,
            num_channels=num_channels,
            img_size=img_size,
        )
    else:
        model = init_model(
            config,
            num_classes=num_classes,
            num_channels=num_channels,
            img_size=img_size,
        )

    model.to(device)

    # Build the loss function
    logging.info("= Loss")
    loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    # Build the optimizer
    logging.info("= Optimizer")
    optim_config = config["optim"]
    optimizer = tl.optim.get_optimizer(optim_config, model.parameters())
    scheduler = tl.optim.get_scheduler(config, optimizer, len(train_loader))

    if config["pretrained"]:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        epoch = checkpoint["epoch"] + 1
    else:
        epoch = 1

    # Configure logging directory
    logdir = configure_logging_directory(log_path, config)
    config["logging"]["logdir"] = str(logdir)

    logging.info(f"Will be logging into {logdir}")

    logdir = pathlib.Path(logdir)
    # Save the config to the logging directory
    with open(logdir / "config.yml", "w") as file:
        yaml.dump(config, file)

    return (
        model,
        optimizer,
        scheduler,
        loss,
        train_loader,
        valid_loader,
        device,
        logdir,
        num_classes,
        ignore_index,
        epoch,
        wandb_log,
        input_size,
        log_path,
    )


def configure_logging_directory(log_path: str, config: dict) -> pathlib.Path:
    """
    Configure the logging directory based on the configuration.
    """
    if not path.isdir(log_path):
        makedirs(log_path)

    if config["pretrained"]:
        logdir = pathlib.Path(log_path)
    else:
        if "wandb" in config["logging"]:
            logdir = log_path + "/" + wandb.run.name
        else:
            logdir = pathlib.Path(utils.generate_unique_logpath(log_path, "UNet"))
        if not path.isdir(logdir):
            makedirs(logdir)

    return logdir


def save_model_summary(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    loss: nn.Module,
    config: dict,
    logdir: pathlib.Path,
    input_size: tuple,
) -> None:
    """
    Save a summary of the model architecture and configuration to the logging directory.
    """
    summary_text = (
        f"Logdir: {logdir}\n"
        "## Command\n"
        f"{' '.join(sys.argv)}\n\n"
        f"Config: {config}\n\n"
        f"Wandb run name: {wandb.run.name}\n\n"
        if config.get("wandb")
        else ""
        "## Summary of the model architecture\n"
        f"{torchinfo.summary(model, input_size=input_size, dtypes=[torch.complex64])}\n\n"
        f"{model}\n\n"
        "## Loss\n\n"
        f"{loss}\n\n"
        "## Datasets:\n"
        f"Train: {train_loader.dataset}\n"
        f"Validation: {valid_loader.dataset}"
    )

    with open(logdir / "summary.txt", "w", encoding="utf-8") as file:
        file.write(summary_text)

    logging.info(summary_text)
    if config.get("wandb"):
        wandb.log({"summary": summary_text})


def retrain(params: list) -> None:
    if len(params) != 1:
        logging.error(f"Usage : {sys.argv[0]} retrain <logdir>")
        sys.exit(-1)

    logdir = pathlib.Path(params[0])
    config_path = logdir / "config.yml"
    logging.info(f"Loading {config_path}")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        config["pretrained"] = True
    train(config)


def train(params: Union[list, dict]) -> None:
    """
    Train the model based on the given configuration.
    """
    if isinstance(params, list):
        if len(params) != 1:
            logging.error(f"Usage : {sys.argv[0]} train <config.yaml>")
            sys.exit(-1)

        logging.info(f"Loading {params[0]}")
        with open(params[0], "r") as file:
            config = yaml.safe_load(file)
    else:
        config = params

    (
        model,
        optimizer,
        scheduler,
        loss,
        train_loader,
        valid_loader,
        device,
        logdir,
        num_classes,
        ignore_index,
        epoch,
        wandb_log,
        input_size,
        log_path,
    ) = load(config)

    # Define the early stopping callback
    model_checkpoint = utils.ModelCheckpoint(
        model, optimizer, logdir, len(input_size), min_is_best=True
    )

    if config["pretrained"]:
        checkpoint_path = (
            log_path + "/best_model.pt"
        )  # to load the last model checkpoint to continue training
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model_checkpoint.best_score = checkpoint["valid_loss"]

    for e in range(epoch, config["nepochs"] + epoch):
        train_metrics = utils.train_epoch(
            model=model,
            loader=train_loader,
            f_loss=loss,
            optim=optimizer,
            scheduler=scheduler,
            device=device,
            number_classes=num_classes,
            epoch=e,
            ignore_index=ignore_index,
        )

        model.eval()
        valid_metrics = utils.valid_epoch(
            model=model,
            loader=valid_loader,
            f_loss=loss,
            device=device,
            number_classes=num_classes,
            ignore_index=ignore_index,
        )
        updated = model_checkpoint.update(epoch=e, score=valid_metrics["valid_loss"])

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(valid_metrics["valid_loss"])
        elif isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]

        logging.info(
            f"[{e}/{config['nepochs'] + epoch}] Valid loss: {round(valid_metrics['valid_loss'],3)} LR: {current_lr:.6f} {'[>> BETTER <<]' if updated else ''}"
        )

        metrics = {**train_metrics, **valid_metrics, "epoch": e}

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            logdir=logdir,
            epoch=e,
            metrics=metrics,
            updated=updated,
        )

        log_metrics(
            wandb_log,
            metrics,
        )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    logdir: pathlib.Path,
    epoch: int,
    metrics: dict,
    updated: bool = False,
) -> None:
    """
    Save model checkpoint if necessary.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": metrics["train_loss"],
        "valid_loss": metrics["valid_loss"],
    }

    torch.save(checkpoint, logdir / "last_model.pt")

    if updated:
        torch.save(checkpoint, logdir / "best_model.pt")


def log_metrics(
    wandb_log: bool,
    metrics: dict,
) -> None:

    if wandb_log:
        logging.info("Logging to WandB")

        # Prepare a dictionary for logging
        log_data = {}
        for key, value in metrics.items():
            if isinstance(value, (list, tuple)) or hasattr(
                value, "__iter__"
            ):  # Check if value is an array-like
                log_data[key] = wandb.Histogram(value)
            else:
                log_data[key] = value

        # Log to WandB
        wandb.log(log_data)

    # Clear CUDA cache
    torch.cuda.empty_cache()


def test(params: list) -> None:
    """
    Test the model based on the given configuration.
    """
    if len(params) != 1:
        logging.error(f"Usage : {sys.argv[0]} test <logdir>")
        sys.exit(-1)

    logdir = pathlib.Path(params[0])
    config_path = logdir / "config.yml"
    logging.info(f"Loading {config_path}")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        config["pretrained"] = True

    log_path = config["logging"]["logdir"]

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    assert config["pretrained"], "No pretrained model available"

    wandb_log = configure_wandb_logging(config)

    seed = config["seed"]
    seed_everything(seed)

    metrics = {}

    checkpoint_path = (
        log_path + "/best_model.pt"
    )  # to load the best model checkpoint for testing
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)

    metrics["Best Loss"] = round(checkpoint["valid_loss"], 5)

    logging.info("= Building the dataloaders")
    data_config = config["data"]
    # data_config["batch_size"] = 1
    (
        train_loader,
        valid_loader,
        test_loader,
        class_weights,
        num_classes,
        num_channels,
        img_size,
        ignore_index,
    ) = dt.get_dataloaders(data_config, use_cuda)

    logging.info("= Model")

    model = load_model(
        checkpoint,
        config,
        num_classes=num_classes,
        num_channels=num_channels,
        img_size=img_size,
    )

    model.eval()
    with torch.no_grad():
        dummy_input = torch.rand(
            (
                config["data"]["batch_size"],
                num_channels,
                img_size,
                img_size,
            ),
            dtype=torch.complex64,
            requires_grad=False,
        )

    logdir = pathlib.Path(log_path)
    logging.info(f"Will be logging into {logdir}")

    res, to_be_vizualized, cm = utils.test_epoch(
        model=model,
        loader=test_loader,
        device=device,
        number_classes=num_classes,
        ignore_index=ignore_index,
    )
    metrics.update(res)

    log_metrics(wandb_log, metrics)

    (
        data_loader,
        nsamples_per_cols,
        nsamples_per_rows,
        indices,
    ) = dt.get_full_image_dataloader(data_config)

    (
        reconstructed_tensors,
        list_of_indices,
    ) = utils.one_forward(
        model=model,
        loader=data_loader,
        device=device,
    )

    image_tensors = []
    ground_truth_tensors = []
    indice_tensors = []

    for data in tqdm.tqdm(data_loader):
        image_tensors.extend(data[0].cpu().detach().numpy())
        ground_truth_tensors.extend(data[1].cpu().detach().numpy())
        indice_tensors.extend(data[2].cpu().detach().numpy())

    ground_truth, sets_masks = dt.reassemble_image(
        segments=ground_truth_tensors,
        samples_per_col=nsamples_per_cols,
        samples_per_row=nsamples_per_rows,
        num_channels=(
            ground_truth_tensors[0].shape[0]
            if len(ground_truth_tensors[0].shape) > 2
            else 1
        ),
        segment_size=config["data"]["img_size"],
        real_indices=indice_tensors,
        sets_indices=indices,
    )

    image_input, _ = dt.reassemble_image(
        segments=image_tensors,
        samples_per_col=nsamples_per_cols,
        samples_per_row=nsamples_per_rows,
        num_channels=(
            image_tensors[0].shape[0] if len(image_tensors[0].shape) > 2 else 1
        ),
        segment_size=config["data"]["img_size"],
        real_indices=indice_tensors,
        sets_indices=None,
    )

    predicted, _ = dt.reassemble_image(
        segments=reconstructed_tensors,
        samples_per_col=nsamples_per_cols,
        samples_per_row=nsamples_per_rows,
        num_channels=(
            reconstructed_tensors[0].shape[0]
            if len(reconstructed_tensors[0].shape) > 2
            else 1
        ),
        segment_size=config["data"]["img_size"],
        real_indices=list_of_indices,
        sets_indices=None,
    )

    to_be_vizualized = [
        image_input[np.newaxis, ...],
        ground_truth[np.newaxis, ...],
        predicted[np.newaxis, ...],
    ]

    vis.plot_segmentation_images(
        to_be_vizualized=to_be_vizualized,
        confusion_matrix=cm,
        number_classes=num_classes,
        ignore_index=ignore_index,
        logdir=logdir,
        wandb_log=wandb_log,
        sets_masks=sets_masks,
    )


if __name__ == "__main__":

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )

    if len(sys.argv) <= 1:
        logging.error(f"Usage : {sys.argv[0]} <train|retrain|test> ...")
        sys.exit(-1)

    command = sys.argv[1]

    eval(f"{command}(sys.argv[2:])")
