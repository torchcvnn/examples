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
from typing import Dict

# External imports
from lightning import Trainer, LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm
from lightning.pytorch.utilities import rank_zero_only


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