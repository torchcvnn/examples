import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import wandb


def plot_segmentation_images(
    to_be_vizualized: list,
    confusion_matrix: np.ndarray,
    number_classes: int,
    logdir: str,
    wandb_log: bool,
    ignore_index: int = None,
    sets_masks: np.ndarray = None,
) -> None:
    """
    Plots segmentation images with an optional test mask overlay to indicate dataset splits.

    Args:
        to_be_vizualized (list): Array of shape (N, 3, H, W), where:
                                       - First channel: Ground truth.
                                       - Second channel: Prediction.
                                       - Third channel: Original image (optional).
        confusion_matrix (np.ndarray): Confusion matrix of shape (number_classes, number_classes).
        number_classes (int): Number of classes for segmentation.
        logdir (str): Directory to save the plot.
        wandb_log (bool): Whether to log the plot to Weights & Biases.
        ignore_index (int, optional): Value in the ground truth to be ignored in the masked prediction.
        sets_masks (np.ndarray, optional): Array of shape (N, H, W) with integer values indicating dataset splits:
                                           - 1: Train
                                           - 2: Validation
                                           - 3: Test
    """
    # Define colormap for segmentation classes
    class_colors = {
        7: {
            0: "black",
            1: "purple",
            2: "blue",
            3: "green",
            4: "red",
            5: "cyan",
            6: "yellow",
        },
        5: {
            0: "black",
            1: "green",
            2: "brown",
            3: "blue",
            4: "yellow",
        },
    }.get(number_classes, {})

    cmap = ListedColormap([class_colors[key] for key in sorted(class_colors.keys())])
    bounds = np.arange(len(class_colors) + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)
    patches = [
        mpatches.Patch(color=class_colors[i], label=f"Class {i}")
        for i in sorted(class_colors.keys())
    ]

    # Define colormap for sets masks
    sets_mask_colors = {
        1: "red",  # Train
        2: "green",  # Validation
        3: "blue",  # Test
    }
    sets_mask_cmap = ListedColormap(
        [sets_mask_colors[key] for key in sorted(sets_mask_colors.keys())]
    )
    sets_mask_bounds = np.arange(len(sets_mask_colors) + 1) - 0.5
    sets_mask_norm = BoundaryNorm(sets_mask_bounds, sets_mask_cmap.N)
    sets_mask_patches = [
        mpatches.Patch(color=sets_mask_colors[i], label=f"Set {i}")
        for i in sorted(sets_mask_colors.keys())
    ]

    # Limit number of samples to visualize
    num_samples = to_be_vizualized[0].shape[0]
    nrows = num_samples + 1  # +1 for confusion matrix
    ncols = 4 if sets_masks is not None else 3  # Add test mask column if available

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 5 * nrows),
        constrained_layout=True,
    )

    # Plot ground truth, predictions, masked predictions, and optionally test masks
    for i in range(num_samples):
        img = to_be_vizualized[0][i]
        g_t = to_be_vizualized[1][i]
        pred = to_be_vizualized[2][i]

        # Mask prediction if ignore_index is provided
        if ignore_index is not None:
            masked_pred = pred.copy()
            masked_pred[g_t == ignore_index] = ignore_index
        else:
            masked_pred = pred

        # Plot ground truth
        g_t = np.squeeze(g_t)
        axes[i][0].imshow(g_t, cmap=cmap, norm=norm, origin="lower")
        axes[i][0].set_title(f"Ground Truth {i+1}")
        axes[i][0].axis("off")

        # Plot prediction
        pred = np.squeeze(pred)
        axes[i][1].imshow(pred, cmap=cmap, norm=norm, origin="lower")
        axes[i][1].set_title(f"Prediction {i+1}")
        axes[i][1].axis("off")

        # Plot masked prediction
        masked_pred = np.squeeze(masked_pred)
        axes[i][2].imshow(masked_pred, cmap=cmap, norm=norm, origin="lower")
        axes[i][2].set_title(f"Masked Prediction {i+1}")
        axes[i][2].axis("off")

        # Plot test mask if available
        if sets_masks is not None:
            axes[i][3].imshow(sets_masks[i], cmap=sets_mask_cmap, norm=sets_mask_norm)
            axes[i][3].set_title(f"Sets Mask {i+1}")
            axes[i][3].axis("off")

    # Plot confusion matrix in the last row
    sns.heatmap(
        confusion_matrix.round(decimals=3),
        annot=True,
        fmt=".2g",
        cmap="Blues",
        ax=axes[-1][0],
        xticklabels=np.setdiff1d(
            np.arange(0, number_classes), np.array([ignore_index])
        ),
        yticklabels=np.setdiff1d(
            np.arange(0, number_classes), np.array([ignore_index])
        ),
    )
    axes[-1][0].set_xlabel("Predicted Class")
    axes[-1][0].set_ylabel("Ground Truth Class")
    axes[-1][0].set_title("Confusion Matrix")

    # Add legends
    legend_ax = axes[-1][1]
    legend_ax.axis("off")
    legend_ax.legend(handles=patches, loc="center", title="Classes")

    # Add sets mask legend if applicable
    if sets_masks is not None:
        test_mask_legend_ax = axes[-1][2]
        test_mask_legend_ax.axis("off")
        test_mask_legend_ax.legend(
            handles=sets_mask_patches, loc="center", title="Test Masks"
        )
    else:
        axes[-1][2].axis("off")

    # Leave extra columns blank for symmetry
    if ncols == 4:
        axes[-1][3].axis("off")

    # Save the figure
    path = f"{logdir}/segmentation_images.png"
    plt.savefig(path, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    # Log to Weights & Biases if enabled
    if wandb_log:
        wandb.log(
            {
                "segmentation_images": [
                    wandb.Image(
                        path, caption="Segmentation Images and Confusion Matrix"
                    )
                ]
            }
        )
