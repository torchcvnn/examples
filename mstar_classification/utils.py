# coding: utf-8
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
from typing import Dict, Sequence, Callable, Any, List
from abc import ABC, abstractmethod
from argparse import ArgumentParser

# External imports
import numpy as np

import torch

from lightning import Trainer, LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm
from lightning.pytorch.utilities import rank_zero_only


def train_parser(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--version', type=int, required=True)
    parser.add_argument('--logdir', type=str, default='training_logs')
    parser.add_argument('--weightdir', type=str, default='weights_storage/vit_mstar')
    parser.add_argument('--datadir', type=str, required=True)
    
    parser.add_argument('--patch_size', type=int, default=7)
    parser.add_argument('--input_size', type=int, default=196)
    parser.add_argument('--batch_size', type=int, default=64)
    
    parser.add_argument('--lr', type=float, default=4e-3)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=10)
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--attention_dropout', type=float, default=0.1)
    parser.add_argument('--norm_layer', type=str, choices=['layer_norm', 'rms_norm'], default='rms_norm')

    return parser



class TBLogger(TensorBoardLogger):
    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float | int], step: int) -> None:
        metrics.pop('epoch', None)
        metrics = {k: v for k, v in metrics.items() if ('step' not in k) and ('val' not in k)}
        super().log_metrics(metrics, step)
    
    
class CustomProgressBar(TQDMProgressBar):
    
    def get_metrics(self, trainer: Trainer, model: LightningModule) -> Dict[str, float]:
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items
    
    def init_train_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for training."""
        bar = super().init_train_tqdm()
        bar.ascii = ' >'
        return bar
    
    def init_validation_tqdm(self) -> Tqdm:
        bar = super().init_validation_tqdm()
        bar.ascii = ' >'
        return bar
    

class complexTransform(ABC):
    
    def __init__(self, always_apply: bool = False, p: float = 0.5) -> None:
        self.always_apply = always_apply
        self.p = p
    
    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    

class ApplyFFT2(complexTransform):
    
    def __init__(self, always_apply: bool = False, p: float = 0.5) -> None:
        super().__init__(always_apply, p)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Apply 2D FFT to the image
        return np.fft.fftshift(np.fft.fft2(image, axes=(0, 1)), axes=(0, 1))
    
    
class ApplyIFFT2(complexTransform):
    
    def __init__(self, always_apply: bool = False, p: float = 0.5) -> None:
        super().__init__(always_apply, p)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Apply 2D IFFT to the image
        return np.fft.ifft2(np.fft.ifftshift(image, axes=(0, 1)), axes=(0, 1))


class PadIfNeeded(complexTransform):
    
    def __init__(self, min_height: int, min_width: int, border_mode: str = 'constant', always_apply: bool = False, p: float = 0.5) -> None:
        super().__init__(always_apply, p)
        
        self.min_height = min_height
        self.min_width = min_width
        self.border_mode = border_mode
        #TODO: other arguments
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Pad the image if it is smaller than the desired size
        image_shapes = image.shape
        pad_top = (self.min_height - image_shapes[0]) // 2
        pad_bottom = self.min_height - image_shapes[0] - pad_top
        pad_left = (self.min_width - image_shapes[1]) // 2
        pad_right = self.min_width - image_shapes[1] - pad_left
        
        paddings = ((pad_top, pad_bottom), (pad_left, pad_right))
        if len(image_shapes) == 3:
            paddings += ((0, 0),)
        return np.pad(
            image,
            paddings,
            mode=self.border_mode
        )
        
        
class Compose(complexTransform):
    
    def __init__(self, transforms: Sequence[Callable], always_apply: bool = False, p: float = 0.5) -> None:
        super().__init__(always_apply, p)
        self.transforms = transforms
        
    def __call__(self, image: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            image = transform(image)
        return image


class ToTensor:
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        # Convert numpy array to PyTorch tensor and Rearrange dimensions from HWC to CHW
        tensor = torch.from_numpy(image).permute(2, 0, 1).to(torch.complex64)
        return tensor