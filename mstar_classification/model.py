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
import embedders

class Model(nn.Module):

    def __init__(self, opt: dict, num_classes: int):
        super().__init__()

        patch_size = opt.patch_size
        input_size = opt.input_size
        embed_dim = opt.embed_dim
        hidden_dim = opt.hidden_dim
        num_layers = opt.num_layers
        num_heads = opt.num_heads
        num_channels = opt.num_channels
        dropout = opt.dropout
        attention_dropout = opt.attention_dropout
        norm_layer = opt.norm_layer

        # The hidden_dim must be adapted to the hidden_dim of the ViT model
        # It is used as the output dimension of the patch embedder but must match
        # the expected hidden dim of your ViT
        num_patches = (input_size // patch_size) ** 2
        embedder = embedders.PatchEmbedderPos(num_patches, 
                                              patch_size, 
                                              embed_dim, 
                                              num_channels, 
                                              dropout)
        # embedder = embedders.PatchEmbedder(input_size, 
        #                                    num_channels, 
        #                                    embed_dim, 
        #                                    patch_size, 
        #                                    norm_layer=norm_layer)

        # For using an off-the shelf ViT model, you can use the following code
        # If you go this way, do not forget to adapt the embed_dim above
        # For vit_t, it is 192
        # self.backbone = c_models.vit_t(embedder)

        # For a custom ViT model, you can use the following code
        # If you go this way, do not forget to adapt the embed_dim above
        # You can reduce it to 32 for example

        self.backbone = c_nn.ViT(
            embedder,
            num_layers,
            num_heads,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm_layer=norm_layer,
        )

        # A Linear decoding head to project on the logits
        self.head = nn.Sequential(
            nn.Linear(embed_dim, num_classes, dtype=torch.complex64)
        )

    def forward(self, x):
        features = self.backbone(x)  # B, num_patches, embed_dim

        # print(features.shape)
    
        # Global average pooling of the patches encoding
        # mean_features = features.mean(dim=1)
        # return self.head(mean_features)

        cls_features = features[:, 0]
        return self.head(cls_features)

class Attention(nn.Module):
    """Complex-valued attention layer for Vision Transformer, as proposed in "Building Blocks for a Complex-Valued Transformer Architecture" by Eilers et al.

    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
    """
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q_norm = c_nn.RMSNorm(self.head_dim)
        self.k_norm = c_nn.RMSNorm(self.head_dim)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, dtype=torch.complex64)

    def forward(self, x: Tensor) -> Tensor:
        B, N, _ = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4) # (3, B, num_heads, num_patches, head_dim)
            .contiguous()
        )
        q, k, v = qkv.unbind(0) # (B, num_heads, num_patches, head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        return self.scaled_dot_product_attention(q, k, v)

    def scaled_dot_product_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        out = ((q @ k.transpose(-2, -1).conj()).real * self.scale).softmax(dim=-1)
        return out.to(torch.complex64) @ v


class Block(nn.Module):
    """Vision Transformer block.

    Args:
        embed_dim (int): Embedded dimension
        hidden_dim (int): Hidden dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate
    """
    def __init__(
        self, embed_dim: int, hidden_dim: int, num_heads: int, dropout: float = 0.0
    ) -> None:
        super().__init__()

        self.attn = Attention(embed_dim, num_heads)
        self.layer_norm = c_nn.RMSNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, dtype=torch.complex64),
            c_nn.CGELU(),
            c_nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim, dtype=torch.complex64),
            c_nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        attn = self.attn(x).transpose(1, 2).reshape(B, N, C)
        x = x + attn
        x = x + self.linear(self.layer_norm(x))
        # inp_x = self.layer_norm(x)
        # x = x + self.attn(inp_x, inp_x, inp_x)[0]
        # x = x + self.linear(self.layer_norm(x))

        return x


class VisionTransformer(nn.Module):
    """Vision Transformer model implementation based on the paper "An image is worth 16x16 words: Transformers for image recognition at scale" by Dosovitskiy et al.
    It is adapted to work with complex-valued inputs, with complex-valued blocks from torchcvnn, and a complex-valued attention layer.

    Args:
        opt (ArgumentParser): model configuration defined in the parser.
        num_classes (int): Number of classes in the dataset.
    """
    # This module was implemented based on 
    def __init__(self, opt: ArgumentParser, num_classes: int) -> None:

        super().__init__()

        patch_size = opt.patch_size
        input_size = opt.input_size
        embed_dim = opt.embed_dim
        hidden_dim = opt.hidden_dim
        num_layers = opt.num_layers
        num_heads = opt.num_heads
        num_channels = opt.num_channels
        dropout = opt.dropout
        attention_dropout = opt.attention_dropout
        # norm_layer = opt.norm_layer
        model_type = opt.model_type

        assert (
            input_size % patch_size == 0
        ), "Image size must be divisible by the patch size"
        num_patches = (input_size // patch_size) ** 2

        # Define whether to use traditional ViT or hybrid-ViT
        # if "hybrid" in model_type:
        #     self.patch_embedder = ConvStem(num_channels, hidden_dim, patch_size)
        #     embed_dim = int(num_channels * (patch_size**2) / 2) # TOCHECK: overwrite embed_dim ?
        #     input_layer_channels = hidden_dim
        # else:
        #     self.patch_embedder = Image2Patch(patch_size)
        #     input_layer_channels = num_channels * (patch_size**2)
        if model_type != "vit":
            raise RuntimeError(f"Model vit-hybrid not reimplmented yet")
        
        self.patch_embedder = embedders.PatchEmbedderPos(num_patches, patch_size, 
                                                         embed_dim, 
                                                         num_channels, dropout)

        # Tranformer blocks
        self.transformer = nn.Sequential(
            *(
                Block(
                    embed_dim, hidden_dim, 
                    num_heads, 
                    dropout=attention_dropout
                )
                for _ in range(num_layers)
            )
        )
        # MLP head
        self.mlp_head = nn.Sequential(
            c_nn.RMSNorm(embed_dim),
            nn.Linear(embed_dim, num_classes, dtype=torch.complex64),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embedder(x)
        x = self.transformer(x)

        cls = x[:, 0] # position of cls_token
        return self.mlp_head(cls)

class BaseClassificationModule(L.LightningModule):
    def __init__(self, opt: ArgumentParser, num_classes: int = 10):
        super().__init__()

        self.opt = opt
        self.ce_loss = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.model = self.configure_model()
        self.gradcam = GradCAM(self.model, "layer3")
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_step_outputs = {}
        self.valid_step_outputs = {}
    
    @staticmethod
    def convert_to_complex(module: nn.Module) -> nn.Module:
        # Patch real-valued architectures into complex-valued ones.
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
                setattr(
                    module,
                    name,
                    c_nn.BatchNorm2d(
                        child.num_features, cdtype=cdtype
                    ),
                )
            elif isinstance(child, nn.MaxPool2d):
                setattr(
                    module,
                    name,
                    c_nn.AvgPool2d(
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
                BaseClassificationModule.convert_to_complex(child)

        return module
        
    @staticmethod
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

    def define_resnet18(self):
        model = resnet18(num_classes=self.num_classes)
        # Modify the first convolutional layer to accept 1 input channel
        model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=model.conv1.out_channels,
            kernel_size=model.conv1.kernel_size,
            stride=model.conv1.stride,
            padding=model.conv1.padding,
            bias=model.conv1.bias is not None,
        )
        return self.convert_to_complex(model)
    
    def define_vit(self):
        return VisionTransformer(self.opt, self.num_classes)
         
    def configure_model(self):
        choices = {"resnet18": self.define_resnet18, "vit": self.define_vit}
        for choice, model_fn in choices.items():
            if choice in self.opt.model_type:
                model = model_fn()
            
        model = nn.Sequential(
            model,
            c_nn.Mod(),
        )
        with torch.no_grad():
            model.apply(self.init_weights)

        return model

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if "resnet18" in self.opt.model_type:
            weight_decay = 0.05
            patience = 5
        else:
            weight_decay = 0.03
            patience = 8
        optimizer = torch.optim.AdamW(
            params=self.parameters(), lr=self.opt.lr, weight_decay=weight_decay
        )
        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer, mode="min", patience=patience, factor=0.5
            ),
            "monitor": "val_loss",  # Metric to monitor
            "interval": "epoch",  # How often to check (epoch or step)
            "frequency": 1,  # Check every epoch
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def plot_gradcam(self, data: Tensor, logger_id: int) -> None:
        assert logger_id < len(self.loggers), "Invalid logger id"
        gradient_cam = self.gradcam(data.repeat(1, 3, 1, 1))
        grid = make_grid(gradient_cam * 0.5 + data * 0.5)  # , make_grid(data)
        self.loggers[logger_id].experiment.add_image(
            "gradcam", grid, self.current_epoch
        )
        # self.loggers[logger_id].experiment.add_image('images', grid[1], self.current_epoch)

    def _training_step(self, data: Tensor, label: Tensor, batch_idx: int) -> Tensor:
        logits = self(data)

        loss = self.ce_loss(logits, label)
        acc = self.accuracy(logits, label)

        self.log("step_loss", loss, prog_bar=True, sync_dist=True)
        self.log("step_metrics", acc, prog_bar=True, sync_dist=True)

        if not self.train_step_outputs:
            self.train_step_outputs = {"step_loss": [loss], "step_metrics": [acc]}
        else:
            self.train_step_outputs["step_loss"].append(loss)
            self.train_step_outputs["step_metrics"].append(acc)

        # if batch_idx == len(self.trainer.train_dataloader) - 1:
        #     self.plot_gradcam(data, 0)

        return loss

    def _validation_step(self, data: Tensor, label: Tensor) -> None:
        logits = self(data)

        loss = self.ce_loss(logits, label)
        acc = self.accuracy(logits, label)
        self.log("step_loss", loss, prog_bar=True, sync_dist=True)
        self.log("step_metrics", acc, prog_bar=True, sync_dist=True)

        if not self.valid_step_outputs:
            self.valid_step_outputs = {"step_loss": [loss], "step_metrics": [acc]}
        else:
            self.valid_step_outputs["step_loss"].append(loss)
            self.valid_step_outputs["step_metrics"].append(acc)

    def _predict_step(self, data: Tensor, label: Tensor) -> Tuple[Tensor]:
        logits = self(data)
        return logits, label

    def on_train_epoch_end(self) -> None:
        _log_dict = {
            "Loss/loss": torch.tensor(self.train_step_outputs["step_loss"]).mean(),
            "Metrics/accuracy": torch.tensor(
                self.train_step_outputs["step_metrics"]
            ).mean(),
        }

        self.loggers[0].log_metrics(_log_dict, self.current_epoch)
        self.train_step_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        mean_loss_value = torch.tensor(self.valid_step_outputs["step_loss"]).mean()
        mean_metrics_value = torch.tensor(
            self.valid_step_outputs["step_metrics"]
        ).mean()

        _log_dict = {
            "Loss/loss": mean_loss_value,
            "Metrics/accuracy": mean_metrics_value,
        }

        self.loggers[1].log_metrics(_log_dict, self.current_epoch)

        self.log("val_loss", mean_loss_value, sync_dist=True)
        self.log("val_Accuracy", mean_metrics_value, sync_dist=True)
        self.valid_step_outputs.clear()


class MSTARClassificationModule(BaseClassificationModule):
    def training_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        data, label = batch
        return super()._training_step(data, label, batch_idx)

    def validation_step(self, batch: List[Tensor], batch_idx: int) -> None:
        data, label = batch
        super()._validation_step(data, label)

    def predict_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        data, label = batch
        return super()._predict_step(data, label)


class SAMPLEClassificationModule(BaseClassificationModule):
    def training_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        data, label, _ = batch
        return super()._training_step(data, label, batch_idx)

    def validation_step(self, batch: List[Tensor], batch_idx: int) -> None:
        data, label, _ = batch
        super()._validation_step(data, label)

    def predict_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        data, label, _ = batch
        return super()._predict_step(data, label)
