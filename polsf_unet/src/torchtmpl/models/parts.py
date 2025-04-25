import torch
import torch.nn as nn
import torch.nn.functional as F
import torchcvnn.nn.modules as c_nn

from functools import partial


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(
        self,
        in_channels,
        out_channels,
        input_size,
        activation,
        normalization_method,
        track_running_stats,
        dtype=torch.complex64,
        dropout=0,
        stride=1,
        mid_channels=None,
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if stride == 1:
            padding = "same"  # padding='same' pads the input so the output has the shape as the input.
            # However, this mode doesn’t support any stride values other than 1.
        else:
            padding = 1

        self.conv1 = nn.Conv2d(
            in_channels,
            mid_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=False,
            padding_mode="circular",
            dtype=dtype,
        )
        if normalization_method == "BatchNorm":
            self.normalization1 = c_nn.BatchNorm2d(
                mid_channels, cdtype=dtype, track_running_stats=track_running_stats
            )
        elif normalization_method == "LayerNorm":
            self.normalization1 = c_nn.LayerNorm(
                normalized_shape=(mid_channels, input_size, input_size)
            )
        elif normalization_method == None:
            self.normalization1 = nn.Identity()

        self.activation = activation
        self.conv2 = nn.Conv2d(
            mid_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=padding,
            bias=False,
            padding_mode="circular",
            dtype=dtype,
        )
        if normalization_method == "BatchNorm":
            self.normalization2 = c_nn.BatchNorm2d(
                num_features=out_channels,
                cdtype=dtype,
                track_running_stats=track_running_stats,
            )
        elif normalization_method == "LayerNorm":
            self.normalization2 = c_nn.LayerNorm(
                normalized_shape=(out_channels, input_size, input_size)
            )
        elif normalization_method == None:
            self.normalization2 = nn.Identity()

        self.dropout = c_nn.Dropout2d(dropout)

        self.shortcut = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                padding=0,
                padding_mode="circular",
                dtype=dtype,
            ),
            self.normalization2,
        )

    def forward(self, x):
        identity = x

        out = self.activation(self.normalization1(self.conv1(x)))
        out = self.normalization2(self.conv2(out))

        out = self.dropout(out)

        identity = self.shortcut(identity)
        out += identity

        out = self.activation(out)
        return out


class Down(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        activation,
        input_size,
        normalization_method,
        track_running_stats,
        downsampling_factor,
        dropout,
        downsampling_method=None,
        stride=1,
    ):
        super().__init__()

        if downsampling_method is None:
            self.downsampling_method = None
        elif downsampling_method == "StridedConv":
            self.downsampling_method = None
            stride = downsampling_factor
        elif downsampling_method == "MaxPool":
            self.downsampling_method = c_nn.MaxPool2d(downsampling_factor)
        elif downsampling_method == "AvgPool":
            self.downsampling_method = c_nn.AvgPool2d(
                downsampling_factor, stride=downsampling_factor
            )

        input_size = input_size // downsampling_factor

        self.conv_layer = DoubleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=activation,
            input_size=input_size,
            normalization_method=normalization_method,
            track_running_stats=track_running_stats,
            stride=stride,
            dropout=dropout,
        )

    def forward(self, x):
        if self.downsampling_method is not None:
            x = self.downsampling_method(x)
        x = self.conv_layer(x)

        return x


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_channels,
        out_channels,
        activation,
        input_size,
        normalization_method,
        track_running_stats,
        upsampling_factor,
        dropout,
        upsampling_method=None,
    ):
        super().__init__()
        if upsampling_method == "Upsample":
            self.upsampling_method = c_nn.Upsample(
                scale_factor=upsampling_factor, mode="bilinear"
            )

        elif upsampling_method == "ConvTranspose":
            self.upsampling_method = c_nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=upsampling_factor
            )
            in_channels = out_channels

        input_size = input_size * upsampling_factor

        in_channels += out_channels

        self.conv_layer = DoubleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=activation,
            input_size=input_size,
            normalization_method=normalization_method,
            track_running_stats=track_running_stats,
            dropout=dropout,
        )

    def forward(self, x1, x2=None):
        x1 = self.upsampling_method(x1)
        x = concat(x1, x2)
        x = self.conv_layer(x)

        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                dtype=torch.complex64,
            ),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


def concat(x1, x2):
    if x2 is None:
        return x1
    else:
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x
