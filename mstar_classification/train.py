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
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable
import os

# External imports
import torch
import torchvision.transforms.v2 as v2
from torchcvnn.datasets import MSTARTargets, SAMPLE
from torchcvnn.transforms import (
    HWC2CHW,
    LogAmplitude,
    ToTensor,
    FFTResize
)
from lightning import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torchmetrics.classification import ConfusionMatrix, Accuracy

import seaborn as sns
import matplotlib.pyplot as plt

# Local imports
from model import MSTARClassificationModule, SAMPLEClassificationModule
from utils import (
    CustomProgressBar,
    TBLogger,
    train_parser,
    get_dataloaders,
    get_datasets
)


def lightning_train_cplxMSTAR(opt: ArgumentParser, trainer: Callable, tmpdir: str) -> None:
    # Dataloading
    dataset = MSTARTargets(
        opt.datadir,
        transform=v2.Compose(
            [
                HWC2CHW(),
                FFTResize((opt.input_size, opt.input_size)),
                LogAmplitude(),
                ToTensor('complex64'),
            ]
        ),
    )
    train_dataset, valid_dataset = get_datasets(dataset)
    train_loader, valid_loader = get_dataloaders(opt, train_dataset, valid_dataset)
    model = MSTARClassificationModule(opt, num_classes=len(dataset.class_names))
    # Train
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    # Predict
    predictions = trainer.predict(dataloaders=valid_loader)
    preds = torch.cat([pred[0].softmax(-1) for pred in predictions], 0)
    labels = torch.cat([label[1] for label in predictions], 0)
    # Plot ConfusionMatrix
    confusion = ConfusionMatrix(task="multiclass", num_classes=len(dataset.class_names))
    confusion.update(preds, labels)
    confusion_matrix = confusion.compute().numpy()
    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)

    plt.figure(figsize=(12.5, 10))
    sns.heatmap(
        confusion_matrix,
        fmt="d",
        cmap="Blues",
        xticklabels=dataset.class_names,
        yticklabels=dataset.class_names,
    )
    plt.savefig(f"{tmpdir}/ConfusionMatrix.png")
    # Top-1 Accuracy
    accuracy_1 = Accuracy(task="multiclass", num_classes=len(dataset.class_names))
    accuracy_1 = accuracy_1(preds, labels)
    print(f"Accuracy top-1: {accuracy_1.item()}")
    # Top-5 Accuracy
    accuracy_2 = Accuracy(
        task="multiclass", num_classes=len(dataset.class_names), top_k=5
    )
    accuracy_2 = accuracy_2(preds, labels)
    print(f"Accuracy top-5: {accuracy_2.item()}")


def lightning_train_cplxSAMPLE(opt: ArgumentParser, trainer: Callable) -> None:
    # Dataloading
    dataset = SAMPLE(
        opt.datadir,
        transform=v2.Compose(
            [
                HWC2CHW(),
                FFTResize((opt.input_size, opt.input_size)),
                LogAmplitude(),
                ToTensor('complex64'),
            ]
        ),
    )
    train_dataset, valid_dataset = get_datasets(dataset)
    train_loader, valid_loader = get_dataloaders(opt, train_dataset, valid_dataset)
    model = SAMPLEClassificationModule(opt, num_classes=len(dataset.class_names))

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    predictions = trainer.predict(dataloaders=valid_loader)
    print(predictions)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = train_parser(parser)
    opt = parser.parse_args()

    tmpdir = os.getenv('TMPDIR', '')
    weightdir = str(tmpdir / Path('weights_storage') / f'version_{opt.version}')
    logdir = str(tmpdir / Path("training_logs") / f"version_{opt.version}")
    trainer = Trainer(
        max_epochs=opt.epochs,
        num_sanity_val_steps=0,
        benchmark=True,
        enable_checkpointing=True,
        callbacks=[
            CustomProgressBar(),
            EarlyStopping(
                monitor="val_loss",
                verbose=True,
                patience=opt.patience,
                min_delta=0.0002,
            ),
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint(
                dirpath=weightdir, monitor="val_Accuracy", verbose=True, mode="max"
            ),
        ],
        logger=[
            TBLogger(logdir, name=None, sub_dir="train", version=opt.version),
            TBLogger(logdir, name=None, sub_dir="valid", version=opt.version),
        ],
    )

    torch.set_float32_matmul_precision("high")
    lightning_train_cplxMSTAR(opt, trainer, tmpdir)
