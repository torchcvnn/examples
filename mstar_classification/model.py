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
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchvision.utils import make_grid
from torchvision.models import resnet18

import lightning as L

from torchmetrics.classification import Accuracy

import torchcvnn.nn as c_nn

from monai.visualize import GradCAM

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

    _norm_layer = {
        'layer_norm': c_nn.LayerNorm,
        'rms_norm': c_nn.RMSNorm,
    }

    def __init__(self, opt: ArgumentParser, num_classes: int = 10):
        super().__init__()

        # The hidden_dim must be adapted to the hidden_dim of the ViT model
        # It is used as the output dimension of the patch embedder but must match
        # the expected hidden dim of your ViT

        embedder = PatchEmbedder(opt.input_size, 1, opt.hidden_dim, opt.patch_size, norm_layer=self._norm_layer[opt.norm_layer])

        # For using an off-the shelf ViT model, you can use the following code
        # If you go this way, do not forget to adapt the hidden_dim above
        # For vit_t, it is 192
        # self.backbone = c_models.vit_t(embedder)

        # For a custom ViT model, you can use the following code
        # If you go this way, do not forget to adapt the hidden_dim above
        # You can reduce it to 32 for example

        mlp_dim = 4 * opt.hidden_dim

        self.backbone = c_nn.ViT(
            embedder,
            opt.num_layers,
            opt.num_heads,
            opt.hidden_dim,
            mlp_dim,
            dropout=opt.dropout,
            attention_dropout=opt.attention_dropout,
            norm_layer=c_nn.LayerNorm,
        )

        # A Linear decoding head to project on the logits
        self.head = nn.Sequential(
            nn.Linear(opt.hidden_dim, num_classes, dtype=torch.complex64), 
            c_nn.Mod()
        )

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)  # B, num_patches, hidden_dim

        # Global average pooling of the patches encoding
        mean_features = features.mean(dim=1)

        return self.head(mean_features)


class Attention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** - 0.5
        self.q_norm = c_nn.RMSNorm(self.head_dim)
        self.k_norm = c_nn.RMSNorm(self.head_dim)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, dtype=torch.complex64)
        
    def forward(self, x: Tensor) -> Tensor:
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        return self.scaled_dot_product_attention(q, k, v)
    
    def scaled_dot_product_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        out = ((q @ k.transpose(-2, -1).conj()).real * self.scale).softmax(dim=-1)
        return out.to(torch.complex64) @ v


class Block(nn.Module):
    def __init__(
        self, 
        embed_dim: int, 
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        
        self.attn = Attention(embed_dim, num_heads)
        self.layer_norm = c_nn.RMSNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, dtype=torch.complex64),
            c_nn.CGELU(),
            c_nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim, dtype=torch.complex64),
            c_nn.Dropout(dropout)
        )
        
    def forward(self, x: Tensor) -> Tensor:
        # x = self.layer_norm(x)
        B, N, C = x.shape
        attn = self.attn(x).transpose(1, 2).reshape(B, N, C)
        x = x + attn
        x = x + self.linear(self.layer_norm(x))
        # inp_x = self.layer_norm(x)
        # x = x + self.attn(inp_x, inp_x, inp_x)[0]
        # x = x + self.linear(self.layer_norm(x))
        
        return x
    

class VisionTransformer(nn.Module):
    def __init__(
        self,
        opt: ArgumentParser,
        num_classes: int
        ) -> None:
        
        super().__init__()
        
        self.patch_size = opt.patch_size
        assert opt.input_size % opt.patch_size == 0, "Image size must be divisible by the patch size"
        self.num_patches = (opt.input_size // opt.patch_size) ** 2
        self.embed_dim = int(opt.num_channels * (opt.patch_size ** 2) / 2)
        self.patch_embedder = ConvStem(opt.num_channels, opt.hidden_dim, opt.patch_size)
        self.input_layer = nn.Linear(opt.hidden_dim, self.embed_dim, dtype=torch.complex64)
        self.transformer = nn.Sequential(
            *(Block(
                self.embed_dim,
                opt.hidden_dim, 
                opt.num_heads,
                dropout=opt.dropout
            ) for _ in range(opt.num_layers))
        )
        self.mlp_head = nn.Sequential(
            c_nn.RMSNorm(self.embed_dim),
            nn.Linear(self.embed_dim, num_classes, dtype=torch.complex64)
        )
        self.dropout = c_nn.Dropout(opt.dropout)
        
        self.cls_token = nn.Parameter(torch.rand(1, 1, self.embed_dim, dtype=torch.complex64))
        self.pos_embedding = nn.Parameter(torch.rand(1, 1 + self.num_patches, self.embed_dim, dtype=torch.complex64))
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embedder(x)
        B, T, _ = x.shape
        x = self.input_layer(x)
        
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding
        
        x = self.dropout(x)
        # x = x.transpose(0, 1)
        x = self.transformer(x)
        
        cls = x[:, 0] # position of cls_token
        return self.mlp_head(cls)


def im_to_patch(im, patch_size, flatten_channels: bool = True):
    B, C, H, W = im.shape
    assert H // patch_size != 0 and W // patch_size != 0, f"Image height and width are {H, W}, which is not a multiple of the patch size"
    
    im = im.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    im = im.permute(0, 2, 4, 1, 3, 5)
    im = im.flatten(1, 2)
    
    if flatten_channels:
        return im.flatten(2, 4)
    else:
        return im
    

class ConvStem(nn.Module):
    def __init__(self, in_channels, hidden_dim, patch_size):
        """
        Convolutional Stem to replace im_to_patch.

        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB images).
            embed_dim (int): The dimensionality of the patch embeddings.
            patch_size (int): The equivalent patch size for downsampling (stride size in the final conv layer).
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, kernel_size=7, stride=2, padding=3, dtype=torch.complex64),
            c_nn.BatchNorm2d(hidden_dim // 2, track_running_stats=False),
            c_nn.modReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=patch_size // 2, padding=1, dtype=torch.complex64),
            c_nn.BatchNorm2d(hidden_dim, track_running_stats=False),
            c_nn.modReLU()
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
        
        Returns:
            torch.Tensor: Patch embeddings of shape (B, embed_dim, H', W').
        """
        x = self.conv(x)  # Apply the convolutional stem
        B, embed_dim, H, W = x.shape
        x = x.flatten(2)  # Flatten the spatial dimensions
        x = x.transpose(1, 2)  # Rearrange to (B, num_patches, embed_dim)
        return x


class BaseResNetModule(L.LightningModule):

    def __init__(self, opt: ArgumentParser, num_classes: int = 10):
        super().__init__()

        self.opt = opt
        self.ce_loss = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.model = self.configure_model()
        self.gradcam = GradCAM(self.model, 'layer3')
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        
        self.train_step_outputs = {}
        self.valid_step_outputs = {}
                
    def configure_model(self):
        model = VisionTransformer(self.opt, self.num_classes)
        model = nn.Sequential(
            model,
            c_nn.Mod(),
        )
        with torch.no_grad():
            model.apply(init_weights)
        
        return model
    
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(params=self.parameters(), lr=self.opt.lr, weight_decay=0.03)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', patience=8, factor=0.5),
            'monitor': 'val_loss',  # Metric to monitor
            'interval': 'epoch',  # How often to check (epoch or step)
            'frequency': 1,  # Check every epoch
        }
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }
        
    def plot_gradcam(self, data: Tensor, logger_id: int) -> None:
        assert logger_id < len(self.loggers), 'Invalid logger id'
        gradient_cam = self.gradcam(data.repeat(1, 3, 1, 1))
        grid = make_grid(gradient_cam * 0.5 + data * 0.5) #, make_grid(data)
        self.loggers[logger_id].experiment.add_image('gradcam', grid, self.current_epoch)
        # self.loggers[logger_id].experiment.add_image('images', grid[1], self.current_epoch)
    
    def _training_step(self, data: Tensor, label: Tensor, batch_idx: int) -> Tensor:
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
            
        # if batch_idx == len(self.trainer.train_dataloader) - 1:
        #     self.plot_gradcam(data, 0)

        return loss

    def _validation_step(self, data: Tensor, label: Tensor) -> None:
        logits = self(data)

        loss = self.ce_loss(logits, label)
        acc = self.accuracy(logits, label)
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
            
    def _predict_step(self, data: Tensor, label: Tensor) -> Tuple[Tensor]:
        logits = self(data)
        return logits, label

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


class ResNetMSTARModule(BaseResNetModule):
    
    def training_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        data, label = batch
        return super()._training_step(data, label, batch_idx)

    def validation_step(self, batch: List[Tensor], batch_idx: int) -> None:
        data, label = batch
        super()._validation_step(data, label)
        
    def predict_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        data, label = batch
        return super()._predict_step(data, label)


class ResNetSAMPLEModule(BaseResNetModule):

    def training_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        data, label, _ = batch
        return super()._training_step(data, label, batch_idx)

    def validation_step(self, batch: List[Tensor], batch_idx: int) -> None:
        data, label, _ = batch
        super()._validation_step(data, label)
        
    def predict_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        data, label, _ = batch
        return super()._predict_step(data, label)

def init_weights(m: nn.Module) -> None:
    """
    Initialize weights for the given module.
    """
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        c_nn.init.complex_kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    if isinstance(m, c_nn.BatchNorm2d):
        m.weight[:, 0, 0] = 1.0
        m.weight[:, 1, 1] = 1.0