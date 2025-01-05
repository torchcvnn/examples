# coding: utf-8

# MIT License

# Copyright (c) 2024 Clément Cornet, Jérémy Fix

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

# External imports
import torch
import torch.nn as nn
import tinycudann as tcnn
import torchcvnn.nn as c_nn


def build_coordinate_2Dt(Nx, Ny, Nt, device=torch.device("cpu")):
    x = torch.linspace(-1, 1, Nx, device=device)
    y = torch.linspace(-1, 1, Ny, device=device)
    t = torch.linspace(-1, 1, Nt, device=device)

    x, y, t = torch.meshgrid(x, y, t, indexing="ij")
    xyt = torch.stack([x, y, t], -1).view(-1, 3)
    xyt = xyt.view(Nx, Ny, Nt, 3)
    return xyt


class ComplexNGP(nn.Module):
    """
    Complex Neural Graphic Primitive

    The first layer involves a Hash Encoding then followed
    by dense layers to get the value of the function at the provided
    input coordinates.

    Arguments:
        n_inputs (int): Number of input coordinates (e.g. 3 for x, y, t)
        n_outputs (int): Number of output values (e.g. 1 for a scalar field)
        encoding_cfg (dict): Configuration for the encoding
        mlp_config (dict): Configuration for the MLP after the encoding
    """

    def __init__(self, n_inputs, n_outputs, encoding_cfg, mlp_cfg):
        super().__init__()

        # The input layer uses a hash encoding
        self.hash_encoder = tcnn.Encoding(n_inputs, encoding_cfg)
        output_hash_encoding = (
            encoding_cfg["n_levels"] * encoding_cfg["n_features_per_level"]
        )

        # And then comes the FFNN with dense layers based
        # on the above coordinate encoding
        n_hidden_units = mlp_cfg["n_hidden_units"]
        n_hidden_layers = mlp_cfg["n_hidden_layers"]
        hidden_activation = c_nn.modReLU
        self.cdtype = torch.complex64

        layers = []
        input_dim = output_hash_encoding
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(input_dim, n_hidden_units, dtype=self.cdtype))
            layers.append(hidden_activation())
            input_dim = n_hidden_units
        # The last dense layer projects onto the output space
        layers.append(nn.Linear(input_dim, n_outputs, dtype=self.cdtype))

        self.ffnn = nn.Sequential(*layers)

    def forward(self, x):
        x = self.hash_encoder(x).to(self.cdtype)
        x = self.ffnn(x)
        return x


def test_ngp():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_inputs = 3
    n_outputs = 4
    model = ComplexNGP(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        encoding_cfg={
            "otype": "Grid",
            "type": "Hash",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hasmap_size": 19,
            "base_resolution": 16,
            "per_level_scale": 2,
            "interpolation": "Linear",
        },
        mlp_cfg={"n_hidden_units": 32, "n_hidden_layers": 2},
    )
    model = model.to(device)

    # Build up the volume over which to evaluate the NIR
    Nx, Ny, Nt = 16, 16, 16
    coords = build_coordinate_2Dt(Nx, Ny, Nt, device=device).view(-1, 3)

    # Sample the volume
    outputs = model(coords).reshape(Nx, Ny, Nt, n_outputs)


if __name__ == "__main__":
    test_ngp()
