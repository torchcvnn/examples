# MIT License
# 
# Copyright (c) 2025 Jérémy Fix, Xuan-Huy Nguyen
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Standard imports
from typing import Tuple, Union, List
from argparse import ArgumentParser

# External imports
import torch
import torch.nn as nn
from torch import Tensor

import lightning as L

from torchmetrics.classification import Accuracy

import torchcvnn.nn as c_nn

# Local imports

class PatchEmbedder(nn.Module):

    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]],
        cin,
        hidden_dim,
        patch_size,
        norm_layer: nn.Module = c_nn.LayerNorm,
        device: torch.device = None,
        dtype=torch.complex64,
    ):
        super(PatchEmbedder, self).__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        self.embedder = nn.Sequential(
            norm_layer([cin, *image_size], **factory_kwargs),
            nn.Conv2d(
                cin,
                hidden_dim,
                kernel_size=patch_size,
                stride=patch_size,
                **factory_kwargs,
            ),
            norm_layer(
                [hidden_dim, image_size[0] // patch_size, image_size[1] // patch_size],
                **factory_kwargs,
            ),
        )

    def rope_embedding(self, H, W, hidden_dim, device):
        """
        Computes and return the 2D rotary positional embedding RoPE from

        "Rotary position embedding for Vision Transformer", Heo et al 2024, ECCV

        Args:
            H (int): Height of the image
            W (int): Width of the image
            hidden_dim (int): Hidden dimension of the model

        Returns:
            torch.Tensor: Positional embeddings for the patches
        """
        # Frequency scale is 10000 in original "Attention is all you need paper"
        # but downscaled to 100 in RoPE paper
        frequency_scale = 100

        pos_H = torch.arange(H, dtype=torch.float32, device=device)
        pos_W = torch.arange(W, dtype=torch.float32, device=device)

        # Compute the positional encoding
        theta_t = frequency_scale ** (
            torch.arange(0, hidden_dim, 2, dtype=torch.float32, device=device)
            / (hidden_dim / 2)
        )

        # Apply the sine/cosine embedding
        emb = torch.exp(1j * theta_t)

        # Compute the positional embedding
        emb_H = emb[:, None] ** pos_H[None, :]  # (hidden_dim//2, H)
        emb_W = emb[:, None] ** pos_W[None, :]  # (hidden_dim//2, W)

        # The even dimensions of the features use the encoding of the height
        # while the odd dimensions use the encoding of the width
        # So that we interleave the emb_H and emb_W
        embeddings = torch.zeros(hidden_dim, H, W, dtype=torch.complex64, device=device)
        embeddings[0::2, :, :] = emb_H[:, :, None]
        embeddings[1::2, :, :] = emb_W[:, None, :]

        return embeddings

    def forward(self, x):
        patch_embeddings = self.embedder(x)  # (B, embed_dim, num_patch_H, num_patch_W)

        num_patches_H, num_patches_W = patch_embeddings.shape[2:]

        # Adds the positionnal embedding
        pos_emb = self.rope_embedding(
            num_patches_H, num_patches_W, patch_embeddings.shape[1], device=x.device
        )
        return patch_embeddings + pos_emb


class Model(nn.Module):

    def __init__(self, opt: ArgumentParser, num_classes: int = 10):
        super().__init__()

        # The hidden_dim must be adapted to the hidden_dim of the ViT model
        # It is used as the output dimension of the patch embedder but must match
        # the expected hidden dim of your ViT
        hidden_dim = 32
        dropout = 0.1
        attention_dropout = 0.1
        # norm_layer = PseudoNorm
        norm_layer = c_nn.RMSNorm
        # norm_layer = c_nn.LayerNorm

        embedder = PatchEmbedder(opt.input_size, 1, hidden_dim, opt.patch_size, norm_layer=norm_layer)

        # For using an off-the shelf ViT model, you can use the following code
        # If you go this way, do not forget to adapt the hidden_dim above
        # For vit_t, it is 192
        # self.backbone = c_models.vit_t(embedder)

        # For a custom ViT model, you can use the following code
        # If you go this way, do not forget to adapt the hidden_dim above
        # You can reduce it to 32 for example

        num_layers = 3
        num_heads = 8
        mlp_dim = 4 * hidden_dim

        self.backbone = c_nn.ViT(
            embedder,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm_layer=c_nn.RMSNorm,
        )

        # A Linear decoding head to project on the logits
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, num_classes, dtype=torch.complex64), 
            c_nn.Mod()
        )

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)  # B, num_patches, hidden_dim

        # Global average pooling of the patches encoding
        mean_features = features.mean(dim=1)

        return self.head(mean_features)
    

class ViTMSTARModel(L.LightningModule):

    def __init__(self, opt: ArgumentParser, num_classes: int = 10):
        super().__init__()

        self.opt = opt
        self.ce_loss = nn.CrossEntropyLoss()
        self.model = Model(opt, num_classes)
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        
        self.train_step_outputs = {}
        self.valid_step_outputs = {}
    
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(params=self.parameters(), lr=self.opt.lr)
    
    def training_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        data, label = batch
        logits = self(data)

        loss = self.ce_loss(logits, label)
        acc = self.accuracy(logits, label)

        self.log('step_loss', loss, prog_bar=True, sync_dist=True)
        self.log('step_metrics', acc, prog_bar=True, sync_dist=True)
        
        if not self.train_step_outputs:
            self.train_step_outputs = {
                'step_loss': [loss],
                'step_metrics': [acc]
            }
        else:
            self.train_step_outputs['step_loss'].append(loss)
            self.train_step_outputs['step_metrics'].append(acc)

        return loss

    def validation_step(self, batch: List[Tensor], batch_idx: int):
        images, labels = batch
        logits = self(images)

        loss = self.ce_loss(logits, labels)
        acc = self.accuracy(logits, labels)
        self.log('step_loss', loss, prog_bar=True, sync_dist=True)
        self.log('step_metrics', acc, prog_bar=True, sync_dist=True)
        
        if not self.valid_step_outputs:
            self.valid_step_outputs = {
                'step_loss': [loss],
                'step_metrics': [acc]
            }
        else:
            self.valid_step_outputs['step_loss'].append(loss)
            self.valid_step_outputs['step_metrics'].append(acc)

    def on_train_epoch_end(self) -> None:
        _log_dict = {
            'Loss/loss': torch.tensor(self.train_step_outputs['step_loss']).mean(),
            'Metrics/accuracy': torch.tensor(self.train_step_outputs['step_metrics']).mean()
        }
        
        self.loggers[0].log_metrics(_log_dict, self.current_epoch)
        self.train_step_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        mean_loss_value = torch.tensor(self.valid_step_outputs['step_loss']).mean()
        mean_metrics_value = torch.tensor(self.valid_step_outputs['step_metrics']).mean()
        
        _log_dict = {
            'Loss/loss': mean_loss_value,
            'Metrics/accuracy': mean_metrics_value
        }
        
        self.loggers[1].log_metrics(_log_dict, self.current_epoch)
        
        self.log('val_loss', mean_loss_value, sync_dist=True)
        self.log('val_Accuracy', mean_metrics_value, sync_dist=True)
        self.valid_step_outputs.clear()