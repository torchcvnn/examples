# coding: utf-8

# MIT License

# Copyright (c) 2026 Jérémy Fix

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
import argparse
import logging
from typing import Callable, Dict, Any
from collections import defaultdict

# External imports
import torch
import torch.nn as nn
import torchcvnn.nn as c_nn

# Local imports
from visualize_stats import visualize_all_statistics

def build_network(input_dim: int, output_dim: int, 
                  num_layers: int, num_hidden: int,
                  scale_factor: float,
                  hidden_activation: Callable):
    layers = []
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(input_dim, num_hidden, 
                                dtype=torch.complex64))
        layers.append(hidden_activation())
        input_dim = num_hidden
        num_hidden = int(num_hidden * scale_factor)

    # The last dense layer projects onto the output space
    layers.append(nn.Linear(input_dim, output_dim,
                            dtype=torch.complex64))
    # Last activation is a Mod to project on R
    layers.append(c_nn.Mod())

    ffnn = nn.Sequential(*layers)
    return ffnn

def init(net: nn.Module):
    for name, module in net.named_modules():
        if hasattr(module, 'weight'):
            c_nn.init.complex_kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

def compute_statistics(tensor: torch.Tensor) -> Dict[str, float]:
    """
    Compute statistics for a tensor.
    
    Args:
        tensor: Input tensor (B, N) or (B, C, H, W)
        
    Returns:
        Dictionary with statistics (mean, std, min, max, norm)
    """
    tensor_flat = tensor.detach().flatten()
    
    stats = {
        'mean': tensor_flat.mean().item(),
        'norm': torch.norm(tensor_flat).mean().item(),
    }
    return stats


def add_activation_gradient_hooks(model: nn.Module) -> Dict[str, Any]:
    """
    Add forward and backward hooks to all modules in the network to collect
    activation and gradient statistics.
    
    Args:
        model: The neural network model to instrument
        
    Returns:
        Dictionary that will be populated with statistics and values during
        forward and backward passes. Structure:
        {
            'layer_name': {
                'activation': {
                    'value': tensor,
                    'stats': {mean, std, min, max, norm}
                },
                'gradient': {
                    'value': tensor,
                    'stats': {mean, std, min, max, norm}
                }
            },
            ...
        }
    """
    stats_dict = defaultdict(dict)
    
    def get_hook_name(module: nn.Module, module_name: str) -> str:
        """Generate a descriptive name for a module."""
        return f"{module_name}({module.__class__.__name__})"
    
    def forward_hook(module, input, output, module_name: str):
        """Hook to capture activation during forward pass."""
        hook_name = get_hook_name(module, module_name)
        
        # Handle different output types
        if isinstance(output, torch.Tensor):
            activation_tensor = output
        elif isinstance(output, tuple):
            activation_tensor = output[0]
        else:
            return
        
        # Store activation value and statistics
        stats_dict['activation'][hook_name] = {
            'value': activation_tensor.detach().clone(),
            'stats': compute_statistics(activation_tensor),
            'shape': tuple(activation_tensor.shape),
        }
    
    def backward_hook(module, grad_input, grad_output, module_name: str):
        """Hook to capture gradients during backward pass."""
        hook_name = get_hook_name(module, module_name)
        
        # grad_output is a tuple of gradients for outputs
        if grad_output and grad_output[0] is not None:
            grad_tensor = grad_output[0]
            
            # Store gradient value and statistics
            stats_dict['gradient'][hook_name] = {
                'value': grad_tensor.detach().clone(),
                'stats': compute_statistics(grad_tensor),
                'shape': tuple(grad_tensor.shape),
            }
    
    # Register hooks for all named modules
    handles = []
    for module_name, module in model.named_modules():
        if module_name == '':  # Skip the root module
            continue
        
        logging.info("Registering hooks for module: %s", get_hook_name(module, module_name))

        # Register forward hook
        forward_handle = module.register_forward_hook(
            lambda m, inp, out, name=module_name: forward_hook(m, inp, out, name)
        )
        handles.append(forward_handle)
        
        # Register backward hook
        backward_handle = module.register_full_backward_hook(
            lambda m, grad_inp, grad_out, name=module_name: backward_hook(m, grad_inp, grad_out, name)
        )
        handles.append(backward_handle)
    
    return stats_dict, handles

def main(args):
    input_dim = args.input_dim
    output_dim = args.output_dim
    num_layers = args.num_layers
    num_hidden = args.num_hidden
    scale_factor = args.scale_factor
    activation = args.activation
    activation_fn = eval(f"c_nn.{activation}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the network
    network = build_network(input_dim, output_dim, 
                            num_layers, num_hidden,
                            scale_factor, activation_fn)
    network = network.to(device)

    # Initialize the network
    init(network)
    
    # Sequence for computing the modulus before feeding the loss function
    loss_function = nn.CrossEntropyLoss()

    # Attach the hooks to collect the statistics
    stats_dict, handles = add_activation_gradient_hooks(network)

    # Generate a dummy input and output tensors
    batch_size = 32
    input_data = torch.randn(batch_size, input_dim, 
                             dtype=torch.complex64, device=device)
    targets = torch.randint(0, output_dim, (batch_size,), device=device)

    # During forward pass
    output = network(input_data)

    # During backward pass
    loss = loss_function(output, targets)
    loss.backward()

    # Access statistics
    logging.info("Activations : ")
    for layer_name, layer_stats in stats_dict['activation'].items():
        print(f"{layer_name} activation mean: {layer_stats['stats']['mean']}")
    logging.info("Gradients : ")
    for layer_name, layer_stats in stats_dict['gradients'].items():
        print(f"{layer_name} gradient norm: {layer_stats['stats']['norm']}")

    # Generate visualizations
    logging.info("\nGenerating visualizations...")
    visualize_all_statistics(stats_dict, output_dir='./plots', show=True)    


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Test to measure the statistics of the activations and gradients, in particular to test initialization functions"
    )
    parser.add_argument(
        "--input_dim",
        default=784, # :)
        type=int,
        help="The input dimensionality",
    )
    parser.add_argument(
        "--num_layers",
        default=10, 
        type=int,
        help="The number of hidden layers",
    )
    parser.add_argument(
        "--num_hidden",
        default=134, 
        type=int,
        help="The number of hidden units for the first layer",
    )
    parser.add_argument(
        "--scale_factor",
        default=1., 
        type=float,
        help="The scale factor the number of hidden units",
    )
    parser.add_argument(
        "--output_dim",
        default=10, # why not :)
        type=int,
        help="The output dimensionality",
    )
    parser.add_argument(
        "--activation",
        default="modReLU",
        type=str,
        choices=["modReLU"],
        help="The activation function",
    )

    args = parser.parse_args()

    main(args)
