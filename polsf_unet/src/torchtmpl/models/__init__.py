# Local imports

import torch
import torch.nn as nn
import torchcvnn.nn as c_nn
from .parts import *
from .model import DeepNeuralNetwork


def build_model(cfg, img_size, num_classes=None, num_channels=None):
    num_layers = cfg["model"]["num_layers"]
    channels_ratio = cfg["model"]["channels_ratio"]
    activation = cfg["model"]["activation"]
    downsampling_method = cfg["model"]["downsampling"]
    upsampling_method = cfg["model"]["upsampling"]
    normalization = cfg["model"]["normalization"]["method"]
    dropout = cfg["model"]["dropout"]
    track_running_stats = cfg["model"]["normalization"]["track_running_stats"]

    assert downsampling_method in [
        "AvgPool",
        "MaxPool",
        "StridedConv",
        None,
    ], "Downsampling method not implemented"
    assert upsampling_method in [
        "ConvTranspose",
        "Upsample",
        None,
    ], "Upsampling method not implemented"
    assert normalization in [
        "BatchNorm",
        "LayerNorm",
        None,
    ], "Normalization method not implemented"

    activation = eval(f"{activation}()", c_nn.__dict__)

    return DeepNeuralNetwork(
        num_channels=num_channels,
        num_classes=num_classes,
        activation=activation,
        input_size=img_size,
        num_layers=num_layers,
        channels_ratio=channels_ratio,
        dropout=dropout,
        normalization_method=normalization,
        track_running_stats=track_running_stats,
        downsampling_method=downsampling_method,
        upsampling_method=upsampling_method,
    )
