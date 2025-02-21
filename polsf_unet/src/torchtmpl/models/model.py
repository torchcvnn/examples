import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from .parts import DoubleConv, Down, Up, OutConv

DOWNSAMPLING_FACTOR = 2
UPSAMPLING_FACTOR = 2


class DeepNeuralNetwork(nn.Module):
    def __init__(
        self,
        num_channels,
        num_layers,
        channels_ratio,
        input_size,
        activation,
        normalization_method,
        track_running_stats,
        downsampling_method,
        upsampling_method,
        dropout,
        num_classes=None,
    ):
        super(DeepNeuralNetwork, self).__init__()

        # Encoder with doubzling channels
        current_channels = channels_ratio
        self.encoder_layers = []
        self.bridge_layers = []
        self.decoder_layers = []

        self.encoder_layers.append(
            DoubleConv(
                in_channels=num_channels,
                out_channels=current_channels,
                activation=activation,
                normalization_method=normalization_method,
                track_running_stats=track_running_stats,
                input_size=input_size,
            )
        )

        for i in range(1, num_layers + 1):
            out_channels = channels_ratio * 2**i
            if i < num_layers:
                self.encoder_layers.append(
                    Down(
                        in_channels=current_channels,
                        out_channels=out_channels,
                        activation=activation,
                        input_size=input_size,
                        normalization_method=normalization_method,
                        track_running_stats=track_running_stats,
                        downsampling_method=downsampling_method,
                        dropout=dropout,
                        downsampling_factor=DOWNSAMPLING_FACTOR,
                    )
                )
            else:
                self.bridge_layers.append(
                    Down(
                        in_channels=current_channels,
                        out_channels=out_channels,
                        activation=activation,
                        input_size=input_size,
                        normalization_method=normalization_method,
                        track_running_stats=track_running_stats,
                        downsampling_method=downsampling_method,
                        dropout=dropout,
                        downsampling_factor=DOWNSAMPLING_FACTOR,
                    )
                )
            input_size //= DOWNSAMPLING_FACTOR
            current_channels = out_channels

        self.encoder_block = nn.Sequential(*self.encoder_layers)

        self.bridge_block = nn.Sequential(*self.bridge_layers)

        # Decoder with halving channels
        for i in range(num_layers - 1, -1, -1):
            out_channels = channels_ratio * 2**i
            self.decoder_layers.append(
                Up(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    activation=activation,
                    input_size=input_size,
                    normalization_method=normalization_method,
                    track_running_stats=track_running_stats,
                    upsampling_method=upsampling_method,
                    dropout=dropout,
                    upsampling_factor=UPSAMPLING_FACTOR,
                )
            )
            input_size *= UPSAMPLING_FACTOR
            current_channels = out_channels

        self.decoder_layers.append(
            OutConv(
                in_channels=current_channels,
                out_channels=num_classes,
            )
        )

        self.decoder_block = nn.Sequential(*self.decoder_layers)

    def forward(self, x):
        list_skip_connections = []

        for enc in self.encoder_block:
            x = enc(x)
            list_skip_connections.append(x)

        x = self.bridge_block(x)

        list_skip_connections = list_skip_connections[::-1]

        for idx, dec in enumerate(self.decoder_block):
            if isinstance(dec, Up):
                skip = list_skip_connections[idx]
                x = dec(x, skip)
            else:
                x = dec(x)

        return x

    def use_checkpointing(self):
        for i, layer in enumerate(self.encoder_block):
            self.encoder_block[i] = checkpoint.checkpoint(layer)
        for i, layer in enumerate(self.bridge_block):
            self.bridge_block[i] = checkpoint.checkpoint(layer)
        for i, layer in enumerate(self.decoder_block):
            self.decoder_block[i] = checkpoint.checkpoint(layer)
