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

# External imports
import torch

import torchvision.transforms.v2 as v2

from torchcvnn.datasets import MSTARTargets, SAMPLE

from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

# Local imports
from model import ResNetMSTARModule, ResNetSAMPLEModule
from utils import (
    CustomProgressBar,
    TBLogger,
    train_parser,
    get_dataloaders,
    get_datasets,
    ApplyFFT2,
    ApplyIFFT2,
    ToTensor,
    CenterCrop,
    LogTransform,
    PadIfNeeded,
    MinMaxNormalize
)

def lightning_train_cplxMSTAR(opt: ArgumentParser, trainer: Trainer):
    # Dataloading
    dataset = MSTARTargets(
        opt.datadir,
        transform=v2.Compose(
            [
                ApplyFFT2(),
                PadIfNeeded(opt.input_size, opt.input_size),
                CenterCrop(opt.input_size, opt.input_size),
                ApplyIFFT2(),
                LogTransform(2e-2, 40),
                ToTensor(),
            ]
        ),
    )
    train_dataset, valid_dataset = get_datasets(dataset)
    train_loader, valid_loader = get_dataloaders(opt, train_dataset, valid_dataset)
    model = ResNetMSTARModule(opt, num_classes=len(dataset.class_names))

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    # predictions = trainer.predict(dataloaders=valid_loader)
    # print(predictions)
    

def lightning_train_cplxSAMPLE(opt: ArgumentParser, trainer: Trainer):
    # Dataloading
    dataset = SAMPLE(
        opt.datadir,
        transform=v2.Compose(
            [
                ApplyFFT2(),
                CenterCrop(opt.input_size, opt.input_size),
                ApplyIFFT2(),
                LogTransform(2e-2, 40),
                ToTensor(),
            ]
        ),
    )
    train_dataset, valid_dataset = get_datasets(dataset)
    train_loader, valid_loader = get_dataloaders(opt, train_dataset, valid_dataset)
    model = ResNetSAMPLEModule(opt, num_classes=len(dataset.class_names))

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    predictions = trainer.predict(dataloaders=valid_loader)
    print(predictions)
    
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser = train_parser(parser)
    opt = parser.parse_args()

    trainer = Trainer(
        max_epochs=opt.epochs,
        num_sanity_val_steps=0,
        benchmark=True,
        enable_checkpointing=True,
        callbacks=[
            CustomProgressBar(),
            EarlyStopping(
                monitor='val_loss', 
                verbose=True,
                patience=opt.patience,
                min_delta=0.0001
            ),
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(
                dirpath=opt.weightdir,
                monitor='val_Accuracy', 
                verbose=True, 
                mode='max'
            )
        ],
        logger=[
            TBLogger(opt.logdir, name=None, sub_dir='train', version=opt.version),
            TBLogger(opt.logdir, name=None, sub_dir='valid', version=opt.version)
        ]
    )
    
    torch.set_float32_matmul_precision('high')
    lightning_train_cplxMSTAR(opt, trainer)