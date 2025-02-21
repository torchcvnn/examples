# MIT License

# Copyright (c) 2025 Xuan-Huy Nguyen

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
import random
from typing import Dict, Tuple
from argparse import ArgumentParser

# External imports
from torch.utils.data import DataLoader, Dataset, Subset

from lightning import Trainer, LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm
from lightning.pytorch.utilities import rank_zero_only


def train_parser(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--datadir", type=str, required=True)

    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--input_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--lr", type=float, default=4e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=15)
    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_channels", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument(
        "--norm_layer", type=str, choices=["layer_norm", "rms_norm"], default="rms_norm"
    )
    parser.add_argument("--model_type", type=str, choices=["resnet18", "vit", "hybrid-vit"], default="hybrid-vit")

    return parser


def get_datasets(dataset: Dataset) -> Tuple[Dataset]:
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    num_valid = int(0.2 * len(dataset))
    train_indices = indices[num_valid:]
    valid_indices = indices[:num_valid]
    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)

    return train_dataset, valid_dataset


def get_dataloaders(
    opt: ArgumentParser, train_dataset: Dataset, valid_dataset: Dataset
) -> Tuple[DataLoader]:
    # Train dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )
    # Validation dataloader
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )
    return train_loader, valid_loader


class TBLogger(TensorBoardLogger):
    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float | int], step: int) -> None:
        metrics.pop("epoch", None)
        metrics = {
            k: v for k, v in metrics.items() if ("step" not in k) and ("val" not in k)
        }
        super().log_metrics(metrics, step)


class CustomProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer: Trainer, model: LightningModule) -> Dict[str, float]:
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items

    def init_train_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for training."""
        bar = super().init_train_tqdm()
        bar.ascii = " >"
        return bar

    def init_validation_tqdm(self) -> Tqdm:
        bar = super().init_validation_tqdm()
        bar.ascii = " >"
        return bar

    def init_predict_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.ascii = " >"
        return bar
