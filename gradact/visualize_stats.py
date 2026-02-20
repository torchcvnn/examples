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
import logging
from typing import Dict, Any, List, Tuple
from collections import defaultdict

# External imports
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import numpy as np
import torch


def extract_layer_stats(stats_dict: Dict[str, Any]) -> Tuple[List[str], Dict[str, Dict[str, list]]]:
    """
    Extract and organize statistics from stats_dict into a structured format.
    
    Args:
        stats_dict: Dictionary containing layer statistics
        
    Returns:
        Tuple of (layer_names, organized_stats)
        where organized_stats has structure:
        {
            'activations': {
                'mean_real': [...],
                'mean_imag': [...],
                'norm': [...]
            },
            'gradients': {
                'mean_real': [...],
                'mean_imag': [...],
                'norm': [...]
            }
        }
    """
    organized_stats = {
        'activations': defaultdict(list),
        'gradients': defaultdict(list)
    }
    
    layer_names = []
    
    # Extract stats in order
    for layer_name in sorted(stats_dict.keys()):
        if layer_name == '_handles':
            continue
            
        layer_names.append(layer_name)
        
        layer_data = stats_dict[layer_name]
        
        # Extract activation statistics
        if 'activation' in layer_data:
            act_stats = layer_data['activation']['stats']
            mean_val = act_stats['mean']
            
            # Handle complex and real means
            if isinstance(mean_val, complex):
                organized_stats['activations']['mean_real'].append(mean_val.real)
                organized_stats['activations']['mean_imag'].append(mean_val.imag)
            else:
                organized_stats['activations']['mean_real'].append(mean_val)
                organized_stats['activations']['mean_imag'].append(0.0)
            
            organized_stats['activations']['norm'].append(act_stats['norm'])
        
        # Extract gradient statistics
        if 'gradient' in layer_data:
            grad_stats = layer_data['gradient']['stats']
            mean_val = grad_stats['mean']
            
            # Handle complex and real means
            if isinstance(mean_val, complex):
                organized_stats['gradients']['mean_real'].append(mean_val.real)
                organized_stats['gradients']['mean_imag'].append(mean_val.imag)
            else:
                organized_stats['gradients']['mean_real'].append(mean_val)
                organized_stats['gradients']['mean_imag'].append(0.0)
            
            organized_stats['gradients']['norm'].append(grad_stats['norm'])
    
    return layer_names, organized_stats


def plot_mean_components(stats_dict: Dict[str, Any], figsize: Tuple[int, int] = (16, 5)):
    """
    Plot real and imaginary components of mean values through layers.
    Separates forward and backward passes into different subplots.
    
    Args:
        stats_dict: Statistics dictionary from the network
        figsize: Figure size (width, height)
    """
    layer_names, organized_stats = extract_layer_stats(stats_dict)
    num_layers = len(layer_names)
    
    if num_layers == 0:
        print("No statistics found in stats_dict")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Mean Value Components through Network Layers', 
                 fontsize=16, fontweight='bold')
    
    layer_indices = np.arange(num_layers)
    
    # Forward pass - Real component
    ax = axes[0, 0]
    if 'mean_real' in organized_stats['activations']:
        act_real = organized_stats['activations']['mean_real']
        ax.plot(layer_indices, act_real, 'o-', linewidth=2.5, markersize=8,
               label='Activations', color='steelblue', alpha=0.8)
        ax.fill_between(layer_indices, act_real, alpha=0.2, color='steelblue')
    
    ax.set_title('Forward Pass - Real Component', fontweight='bold', fontsize=12)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Re(Mean)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_xticks(layer_indices[::max(1, num_layers // 8)])
    
    # Forward pass - Imaginary component
    ax = axes[0, 1]
    if 'mean_imag' in organized_stats['activations']:
        act_imag = organized_stats['activations']['mean_imag']
        ax.plot(layer_indices, act_imag, 'o-', linewidth=2.5, markersize=8,
               label='Activations', color='steelblue', alpha=0.8)
        ax.fill_between(layer_indices, act_imag, alpha=0.2, color='steelblue')
    
    ax.set_title('Forward Pass - Imaginary Component', fontweight='bold', fontsize=12)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Im(Mean)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_xticks(layer_indices[::max(1, num_layers // 8)])
    
    # Backward pass - Real component
    ax = axes[1, 0]
    if 'mean_real' in organized_stats['gradients']:
        grad_real = organized_stats['gradients']['mean_real']
        ax.plot(layer_indices, grad_real, 's-', linewidth=2.5, markersize=8,
               label='Gradients', color='darkred', alpha=0.8)
        ax.fill_between(layer_indices, grad_real, alpha=0.2, color='darkred')
    
    ax.set_title('Backward Pass - Real Component', fontweight='bold', fontsize=12)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Re(Mean)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_xticks(layer_indices[::max(1, num_layers // 8)])
    
    # Backward pass - Imaginary component
    ax = axes[1, 1]
    if 'mean_imag' in organized_stats['gradients']:
        grad_imag = organized_stats['gradients']['mean_imag']
        ax.plot(layer_indices, grad_imag, 's-', linewidth=2.5, markersize=8,
               label='Gradients', color='darkred', alpha=0.8)
        ax.fill_between(layer_indices, grad_imag, alpha=0.2, color='darkred')
    
    ax.set_title('Backward Pass - Imaginary Component', fontweight='bold', fontsize=12)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Im(Mean)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_xticks(layer_indices[::max(1, num_layers // 8)])
    
    plt.tight_layout()
    return fig


def plot_norm_values(stats_dict: Dict[str, Any], figsize: Tuple[int, int] = (12, 5)):
    """
    Plot norm values through layers.
    
    Args:
        stats_dict: Statistics dictionary from the network
        figsize: Figure size (width, height)
    """
    layer_names, organized_stats = extract_layer_stats(stats_dict)
    num_layers = len(layer_names)
    
    if num_layers == 0:
        print("No statistics found in stats_dict")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    cmap_forward = cm.get_cmap('Blues')
    cmap_backward = cm.get_cmap('Reds')
    
    norm_forward = Normalize(vmin=0, vmax=num_layers - 1)
    norm_backward = Normalize(vmin=0, vmax=num_layers - 1)
    
    layer_indices = np.arange(num_layers)
    x_offset = 0.2
    width = 0.35
    
    # Plot activation norms as bars
    if 'norm' in organized_stats['activations']:
        act_norms = organized_stats['activations']['norm']
        colors = [cmap_forward(norm_forward(i)) for i in range(num_layers)]
        ax.bar(layer_indices - x_offset, act_norms, width, label='Forward (Activations)',
               color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Plot gradient norms as bars
    if 'norm' in organized_stats['gradients']:
        grad_norms = organized_stats['gradients']['norm']
        colors = [cmap_backward(norm_backward(num_layers - 1 - i)) for i in range(num_layers)]
        ax.bar(layer_indices + x_offset, grad_norms, width, label='Backward (Gradients)',
               color='darkred', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_title('Norm of Values through Network Layers', fontweight='bold', fontsize=14)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Norm')
    ax.set_xticks(layer_indices)
    ax.set_xticklabels([f'{i}' for i in range(num_layers)])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    return fig


def plot_complex_scatter(stats_dict: Dict[str, Any], figsize: Tuple[int, int] = (16, 8)):
    """
    Plot scatter distributions of complex numbers in 2D space.
    Ensures consistent x/y ranges across layers within the same pass.
    
    Args:
        stats_dict: Statistics dictionary from the network
        figsize: Figure size (width, height)
    """
    num_layers = len(stats_dict['activation'])
    
    # Calculate global range for activations (forward pass)
    act_max = None
    activations_dict = stats_dict['activation']
    for idx, (layer_name, act_stats) in enumerate(activations_dict.items()):
        values = act_stats['value']
        if values is not None and len(values) > 0:
            if torch.is_complex(values):
                maxi = max(values.real.abs().max().item(),
                           values.imag.abs().max().item())
            else:
                maxi = values.abs().max().item()

            if act_max is None:
                act_max = maxi
            else:
                act_max = max(act_max, maxi)
    logging.info(f"Act max : {act_max}")
    if act_max is None:
        act_max = 1.0
    else:
        act_max *= 1.2

    # Calculate global range for gradients (backward pass)
    grad_max = None
    gradivations_dict = stats_dict['gradient']
    for idx, (layer_name, grad_stats) in enumerate(gradivations_dict.items()):
        values = grad_stats['value']
        if values is not None and len(values) > 0:
            if torch.is_complex(values):
                maxi = max(values.real.abs().max().item(),
                           values.imag.abs().max().item())
            else:
                maxi = values.abs().max().item()

            if grad_max is None:
                grad_max = maxi
            else:
                grad_max = max(grad_max, maxi)
    logging.info(f"Grad max : {grad_max}")

    if grad_max is None:
        grad_max = 1.0
    else:
        grad_max *= 1.2
    
    # Determine grid layout
    cols = min(5, max(1, num_layers // 2))
    rows = 2
    # Determine the indices of the layers to plot
    layer_indices = np.round(np.linspace(0, num_layers - 1, cols)).astype(int)
    logging.info(f"Going to show the layers of indices : {layer_indices}")
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols) if axes.ndim == 1 else axes
    
    fig.suptitle('Complex Value Distributions through Layers\n(Forward & Backward Passes)', 
                 fontsize=16, fontweight='bold')
    
    cmap_forward = cm.get_cmap('Blues')
    cmap_backward = cm.get_cmap('Reds')
    
    norm_forward = Normalize(vmin=0, vmax=num_layers - 1)
    norm_backward = Normalize(vmin=0, vmax=num_layers - 1)
    
    # Plot activations (forward pass)
    activations_list = list(stats_dict['activation'].items())
    for idx, lidx in enumerate(layer_indices):
        layer_name = activations_list[lidx][0]
        act_stats = activations_list[lidx][1]
        values = act_stats["value"].cpu().numpy()
        if idx >= cols:
            break
        ax = axes[0, idx]
        
        if values is not None and len(values) > 0:
            real_parts = np.real(values)
            imag_parts = np.imag(values)
            
            color = cmap_forward(norm_forward(idx))
            ax.scatter(real_parts, imag_parts, alpha=0.5, s=20, color=color, edgecolors='black', linewidth=0.3)
            
            # Use global range for all activation plots
            ax.set_xlim(-act_max, act_max)
            ax.set_ylim(-act_max, act_max)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Real', fontsize=9)
            ax.set_ylabel('Imag', fontsize=9)
            ax.set_title(f'{layer_name[:15]}...\n(Forward)', fontsize=9, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_visible(False)
    
    # Plot gradients (backward pass)
    gradients_list = list(stats_dict['gradient'].items())[::-1]
    for idx, lidx in enumerate(layer_indices):
        layer_name = gradients_list[lidx][0]
        grad_stats = gradients_list[lidx][1]
        values = grad_stats["value"].cpu().numpy()
        if idx >= cols:
            break
        ax = axes[1, idx]
        
        if values is not None and len(values) > 0:
            real_parts = np.real(values)
            imag_parts = np.imag(values)
            
            # Use reversed color for backward pass
            color = cmap_backward(norm_backward(num_layers - 1 - idx))
            ax.scatter(real_parts, imag_parts, alpha=0.5, s=20, color=color, edgecolors='black', linewidth=0.3)
            
            # Use global range for all gradient plots
            ax.set_xlim(-grad_max, grad_max)
            ax.set_ylim(-grad_max, grad_max)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Real', fontsize=9)
            ax.set_ylabel('Imag', fontsize=9)
            ax.set_title(f'{layer_name[:15]}...\n(Backward)', fontsize=9, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_visible(False)
    
    # Hide unused subplots
    for idx in range(num_layers, cols):
        axes[0, idx].set_visible(False)
        axes[1, idx].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_mean_complex_trajectory(stats_dict: Dict[str, Any], figsize: Tuple[int, int] = (14, 6)):
    """
    Plot the trajectory of mean values in complex plane as they progress through layers.
    Separates forward and backward passes into different subplots for better visualization
    when they have different scales. Axes are centered on (0, 0) with symmetric ranges.
    
    Args:
        stats_dict: Statistics dictionary from the network
        figsize: Figure size (width, height)
    """
    layer_names, organized_stats = extract_layer_stats(stats_dict)
    num_layers = len(layer_names)
    
    if num_layers == 0:
        print("No statistics found in stats_dict")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Mean Value Trajectories in Complex Plane', fontweight='bold', fontsize=16)
    
    cmap_forward = cm.get_cmap('Blues')
    norm_forward = Normalize(vmin=0, vmax=num_layers - 1)
    
    # Plot forward pass (activations)
    ax = axes[0]
    if 'mean_real' in organized_stats['activations'] and 'mean_imag' in organized_stats['activations']:
        act_real = organized_stats['activations']['mean_real']
        act_imag = organized_stats['activations']['mean_imag']
        
        # Plot points with color gradient
        for i in range(num_layers - 1):
            color = cmap_forward(norm_forward(i))
            ax.plot(act_real[i:i+2], act_imag[i:i+2], 'o-', linewidth=2.5, markersize=8,
                   color=color, alpha=0.8)
        
        # Add layer index labels
        for i, (real, imag) in enumerate(zip(act_real, act_imag)):
            ax.text(real, imag, f'  {i}', fontsize=8, ha='left')
        
        ax.scatter(act_real, act_imag, s=100, c=range(num_layers), cmap='Blues',
                  edgecolors='black', linewidth=1, alpha=0.8, zorder=5)
        
        # Center on (0, 0) with symmetric range
        max_val = max(np.abs(act_real).max(), np.abs(act_imag).max())
        lim = max_val * 1.2
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    
    ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
    ax.set_title('Forward Pass (Activations)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Real Part', fontsize=11)
    ax.set_ylabel('Imaginary Part', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Plot backward pass (gradients)
    cmap_backward = cm.get_cmap('Reds')
    norm_backward = Normalize(vmin=0, vmax=num_layers - 1)
    
    ax = axes[1]
    if 'mean_real' in organized_stats['gradients'] and 'mean_imag' in organized_stats['gradients']:
        grad_real = organized_stats['gradients']['mean_real']
        grad_imag = organized_stats['gradients']['mean_imag']
        
        # Plot points with reversed color gradient
        for i in range(num_layers - 1):
            color = cmap_backward(norm_backward(num_layers - 1 - i))
            ax.plot(grad_real[i:i+2], grad_imag[i:i+2], 's-', linewidth=2.5, markersize=8,
                   color=color, alpha=0.8)
        
        # Add layer index labels
        for i, (real, imag) in enumerate(zip(grad_real, grad_imag)):
            ax.text(real, imag, f'  {i}', fontsize=8, ha='left')
        
        ax.scatter(grad_real, grad_imag, s=100, c=range(num_layers), cmap='Reds',
                  edgecolors='black', linewidth=1, alpha=0.8, zorder=5)
        
        # Center on (0, 0) with symmetric range
        max_val = max(np.abs(grad_real).max(), np.abs(grad_imag).max())
        lim = max_val * 1.2
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    
    ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
    ax.set_title('Backward Pass (Gradients)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Real Part', fontsize=11)
    ax.set_ylabel('Imaginary Part', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    return fig


def visualize_all_statistics(stats_dict: Dict[str, Any], output_dir: str = '.', show: bool = True):
    """
    Create and save all visualization plots.
    
    Args:
        stats_dict: Statistics dictionary from the network
        output_dir: Directory to save plots
        show: Whether to show the plots interactively
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create all plots
    # fig1 = plot_mean_components(stats_dict)
    # if fig1:
    #     fig1.savefig(os.path.join(output_dir, '01_mean_components.png'), dpi=150, bbox_inches='tight')
    #     print(f"Saved: {os.path.join(output_dir, '01_mean_components.png')}")
    
    # fig2 = plot_norm_values(stats_dict)
    # if fig2:
    #     fig2.savefig(os.path.join(output_dir, '02_norm_values.png'), dpi=150, bbox_inches='tight')
    #     print(f"Saved: {os.path.join(output_dir, '02_norm_values.png')}")
    
    # fig3 = plot_mean_complex_trajectory(stats_dict)
    # if fig3:
    #     fig3.savefig(os.path.join(output_dir, '03_mean_complex_trajectory.png'), dpi=150, bbox_inches='tight')
    #     print(f"Saved: {os.path.join(output_dir, '03_mean_complex_trajectory.png')}")
    
    fig4 = plot_complex_scatter(stats_dict)
    if fig4:
        fig4.savefig(os.path.join(output_dir, '04_complex_value_distributions.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: {os.path.join(output_dir, '04_complex_value_distributions.png')}")
    
    if show:
        plt.show()

