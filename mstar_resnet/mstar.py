# coding: utf-8

# MIT License

# Copyright (c) 2024 Jérémy Fix

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

"""
# Example using complex valued neural networks to classify the SAMPLE data

In this example, we will use the complex valued neural networks to classify the SAMPLE data. This sample script also shows how to patch a pre-constructed neural network, as provided by TIMM to make it complex valued.

We benefit from timm to build the architecture but then replace the real valued modules by complex valued counterparts.

Requires dependencies :
    python3 -m pip install torchcvnn timm torchvision
"""

# Standard imports
import random
import argparse
import logging
import sys

# External imports
import torch
import torch.nn as nn
from torchvision.transforms import v2
import timm
import numpy as np

# Local imports
import torchcvnn
import torchcvnn.nn as c_nn
import torchcvnn.datasets
import utils

MIN_VALUE = 0.02
MAX_VALUE = 40


class LogAmplitudeTransform:
    def __init__(self):
        pass

    def __call__(self, tensor) -> torch.Tensor:
        new_tensor = self._transform_amplitude(tensor)
        return new_tensor

    def _transform_amplitude(self, tensor: torch.Tensor) -> torch.Tensor:
        new_tensor = []
        for idx, ch in enumerate(tensor):
            amplitude = torch.abs(ch)
            phase = torch.angle(ch)
            min_val = MIN_VALUE
            max_val = MAX_VALUE
            amplitude = torch.clip(amplitude, min_val, max_val)
            transformed_amplitude = (
                torch.log10(amplitude) - torch.log10(torch.tensor([min_val]))
            ) / (
                torch.log10(torch.tensor([max_val]))
                - torch.log10(torch.tensor([min_val]))
            )
            new_tensor.append(transformed_amplitude * torch.exp(1j * phase))
        return torch.as_tensor(np.stack(new_tensor), dtype=torch.complex64)


def get_dataloaders(datadir, batch_size=64, valid_ratio=0.1):
    transform = v2.Compose(
        transforms=[
            v2.ToImage(),
            v2.Resize(128),
            v2.CenterCrop(128),
            LogAmplitudeTransform(),
        ]
    )

    train_valid_dataset = torchcvnn.datasets.SAMPLE(
        datadir, transform=transform, download=True
    )

    all_indices = list(range(len(train_valid_dataset)))
    random.shuffle(all_indices)
    split_idx = int(valid_ratio * len(train_valid_dataset))
    valid_indices, train_indices = all_indices[:split_idx], all_indices[split_idx:]

    # Train dataloader
    train_dataset = torch.utils.data.Subset(train_valid_dataset, train_indices)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # Valid dataloader
    valid_dataset = torch.utils.data.Subset(train_valid_dataset, valid_indices)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False
    )

    num_classes = len(train_valid_dataset.class_names)

    return train_loader, valid_loader, num_classes


def convert_to_complex(module: nn.Module) -> nn.Module:
    cdtype = torch.complex64
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(
                module,
                name,
                nn.Conv2d(
                    child.in_channels,
                    child.out_channels,
                    child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    bias=child.bias is not None,
                    dtype=cdtype,
                ),
            )

        elif isinstance(child, nn.ReLU):
            setattr(module, name, c_nn.modReLU())

        elif isinstance(child, nn.BatchNorm2d):
            setattr(module, name, c_nn.BatchNorm2d(child.num_features))

        elif isinstance(child, nn.MaxPool2d):
            setattr(
                module,
                name,
                c_nn.MaxPool2d(
                    child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                ),
            )
        elif isinstance(child, nn.Linear):
            setattr(
                module,
                name,
                nn.Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    dtype=cdtype,
                ),
            )
        else:
            convert_to_complex(child)

    return module


def init_weights(m: nn.Module) -> None:
    """
    Initialize weights for the given module.
    """
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        c_nn.init.complex_kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def train(datadir):
    """
    Train function

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_ratio = 0.3
    batch_size = 128
    epochs = 100
    lr = 0.001

    # Dataloading
    train_loader, valid_loader, num_classes = get_dataloaders(
        datadir, batch_size=batch_size, valid_ratio=valid_ratio
    )

    X, _, _ = next(iter(train_loader))
    in_chans = X.shape[1]

    # Build the model as a patched TIMM
    # and send it to the right device
    real_valued_model = timm.create_model(
        "resnet50", pretrained=False, num_classes=num_classes, in_chans=in_chans
    )
    model = convert_to_complex(real_valued_model)
    # Add a final layer to the model to transform the complex valued logits into
    # real valued logits to go into the CrossEntropyLoss
    model = nn.Sequential(
        model,
        c_nn.Mod(),
    )  # not sure if this is the right way to do it

    # Initialize the weights
    model.apply(init_weights)

    model.to(device)

    # Loss, optimizer, callbacks
    f_loss = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)
    logpath = utils.generate_unique_logpath("./logs", "SAMPLE")
    logging.info(f"Logging to {logpath}")
    checkpoint = utils.ModelCheckpoint(model, logpath, 4, min_is_best=True)

    # Training loop
    for e in range(epochs):
        logging.info(">> Training")
        train_loss, train_acc = utils.train_epoch(
            model, train_loader, f_loss, optim, device
        )

        logging.info(">> Testing")
        valid_loss, valid_acc = utils.test_epoch(model, valid_loader, f_loss, device)
        updated = checkpoint.update(valid_loss)
        scheduler.step()
        better_str = "[>> BETTER <<]" if updated else ""

        logging.info(
            f"[Step {e}] Train : CE {train_loss:5.2f} Acc {train_acc:5.2f} | Valid : CE {valid_loss:5.2f} Acc {valid_acc:5.2f}"
            + better_str
            + f" | LR {scheduler.get_last_lr()[0]:.5f}"
        )


if __name__ == "__main__":

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )
    parser = argparse.ArgumentParser(description="SAMPLE classification with torchcvnn")
    parser.add_argument(
        "--datadir", type=str, default="data", help="Path to the data directory"
    )

    args = parser.parse_args()

    datadir = args.datadir

    train(datadir)
