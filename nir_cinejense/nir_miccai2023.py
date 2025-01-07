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

# Standard imports
import argparse
import logging
import random
import pathlib
import shutil

# External imports
import tqdm
import torch
import torch.nn as nn
import numpy as np
import matplotlib
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

matplotlib.use("Agg")  # Non interactive backend

import matplotlib.pyplot as plt

# Local imports
from torchcvnn.datasets.miccai2023 import (
    MICCAI2023,
    CINEView,
    kspace_to_image,
    AccFactor,
    combine_coils_from_kspace,
)
import utils


@torch.jit.script
def FFT(x):
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(x, dim=(0, 1)), dim=(0, 1)), dim=(0, 1)
    )


@torch.jit.script
def IFFT(x):
    return torch.fft.ifftshift(
        torch.fft.ifft2(torch.fft.fftshift(x, dim=(0, 1)), dim=(0, 1)), dim=(0, 1)
    )


def combine_coils(kspace):
    """
    Combine the coils from the given k-space

    Arguments:
        kspace: Tensor of shape (nrows, ncols, ncoils)
                     or (nrows, ncols, ncoils, nframes)
                complex valued

    Returns:
        image: Tensor of shape (nrows, ncols)
                    or (nrows, ncols, nframes)
                magnitude only
    """
    if isinstance(kspace, np.ndarray):
        kspace = torch.tensor(kspace, dtype=torch.complex64)

    images = IFFT(kspace)

    # Combine the coils in the image space with the RSS
    coils_combined = (images.abs() ** 2).sum(axis=2).sqrt()

    return coils_combined


class TVLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, X):
        """
        Computes the total variation of the tensor X

        X is of shape (nrows, ncols, nframes, ncoils)

        The TV loss is computed from the 2D signals nrows x ncols
        averaged over the other dimensions
        """

        diff_rows = X[1:, :, ...] - X[:-1, :, ...]
        diff_cols = X[:, 1:, ...] - X[:, :-1, ...]
        tv = torch.mean(torch.abs(diff_rows) ** 2) + torch.mean(
            torch.abs(diff_cols) ** 2
        )
        return tv


def infer_on_slice(
    subsampled_slice, subsampled_mask, slice_idx, results_dir, training_cfg
):
    """
    Perform inference on a single slice for all the frames and all the coils

    Arguments:
        subsampled_slice (torch.Tensor): Subsampled k-space data for a single slice, (ky, kx, sc, t)
        subsampled_mask (torch.Tensor): Subsampled mask for a single slice (ky, kx)
        slice_idx (int): the index of the slice, used to determine the results filenames
        results_dir (pathlib.Path): the path where to save all the results
        training_cfg (dict): the training parameters (reg_weight, learning rate, number of iterations)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reg_weight = training_cfg["reg_weight"]
    max_iter = training_cfg["iter"]
    lr = training_cfg["lr"]

    # Put the slices on the right device
    subsampled_slice = torch.tensor(subsampled_slice, dtype=torch.complex64).to(device)
    subsampled_mask = torch.tensor(subsampled_mask, dtype=torch.float32).to(device)

    # Build the models
    nrows, ncols, ncoils, nframes = subsampled_slice.shape

    # Pre-compute the coordinates for sampling the 3D volume X, Y, T
    # and view it as (n_samples, 3)
    coor = utils.build_coordinate_2Dt(nrows, ncols, nframes, device).view(-1, 3)

    # Build the image model
    encoding_image = {
        "otype": "Grid",
        "type": "Hash",
        "n_levels": 16,
        "n_features_per_level": 2,
        "log2_hasmap_size": 19,
        "base_resolution": 16,
        "per_level_scale": 2,
        "interpolation": "Linear",
    }
    mlp_image = {"n_hidden_units": 32, "n_hidden_layers": 4}
    image_model = utils.ComplexNGP(
        n_inputs=3, n_outputs=1, encoding_cfg=encoding_image, mlp_cfg=mlp_image
    ).to(device)

    # Build the Coil Sensitivity Map network
    encoding_csm = {
        "otype": "Grid",
        "type": "Hash",
        "n_levels": 4,
        "n_features_per_level": 8,
        "log2_hasmap_size": 19,
        "base_resolution": 2,
        "per_level_scale": 1.1,
        "interpolation": "Linear",
    }
    mlp_csm = {"n_hidden_units": 32, "n_hidden_layers": 4}
    csm_model = utils.ComplexNGP(
        n_inputs=3, n_outputs=ncoils, encoding_cfg=encoding_csm, mlp_cfg=mlp_csm
    ).to(device)

    # Built the optimizers and losses
    optim_image = torch.optim.Adam(image_model.parameters(), lr=lr)
    optim_csm = torch.optim.Adam(csm_model.parameters(), lr=lr)

    # The loss has two components.
    #  - A Total Variation Loss in the image space for the pre-intensity
    #  - a Huber loss between the sampled components of the k-space
    reg_loss = TVLoss()
    kspace_loss = torch.nn.HuberLoss()

    # Loop for max_iter
    with tqdm.tqdm(range(max_iter)) as pbar:
        for _ in pbar:

            # Switch the models in training mode
            image_model.train()
            csm_model.train()

            # Compute the forward pass
            pre_intensity = image_model(coor).view(
                nrows, ncols, nframes
            )  # Nrows, Ncols, Nframes
            csm = csm_model(coor).view(
                nrows, ncols, nframes, ncoils
            )  # Nrows, Ncols, Nframes, Ncoils

            # Compute the RSS over the coils
            csm_norm = torch.sqrt((csm.conj() * csm).sum(axis=-1))
            # Unsqueeze over the coil dimension to apply the same scaling for every coil
            csm = csm / (csm_norm.unsqueeze(-1) + 1e-12)

            # Apply the same pre-instensity through every coil specific sensitivity
            fft_pre_intensity = FFT(pre_intensity.unsqueeze(axis=-1) * csm).transpose(
                3, 2
            )  # (Nrows, Ncols, Ncoils, Nframes)

            # Compute the loss with the reconstruction loss
            # and the regularization loss
            masked_pred_kspace = torch.view_as_real(
                fft_pre_intensity[subsampled_mask == 1]
            )
            masked_kspace = torch.view_as_real(subsampled_slice[subsampled_mask == 1])

            kspace_loss_value = kspace_loss(masked_pred_kspace, masked_kspace)
            reg_loss_value = reg_loss(pre_intensity)
            loss = kspace_loss_value + reg_weight * reg_loss_value
            pbar.set_postfix(
                {
                    "TV": reg_loss_value.item(),
                    "Data consistency Loss": kspace_loss_value.item(),
                }
            )

            # Zero grad, backward and update
            optim_image.zero_grad()
            optim_csm.zero_grad()

            loss.backward()

            optim_image.step()
            optim_csm.step()

    # Inference
    logging.info("Performing inference")
    image_model.eval()
    csm_model.eval()
    with torch.no_grad():
        pre_intensity = image_model(coor).view(
            nrows, ncols, nframes
        )  # Nrows, Ncols, Nframes
        csm = csm_model(coor).view(
            nrows, ncols, nframes, ncoils
        )  # Nrows, Ncols, Nframes, Ncoils

        # Compute the RSS over the coils
        csm_norm = torch.sqrt((csm.conj() * csm).sum(axis=-1))
        # Unsqueeze over the coil dimension to apply the same scaling for every coil
        csm = csm / (csm_norm.unsqueeze(-1) + 1e-12)

        fft_pre_intensity = FFT(
            pre_intensity.unsqueeze(axis=-1) * csm
        )  # (Nrows, Ncols, Nframes, Ncoils)

        recon_kspace = torch.clone(fft_pre_intensity)
        # Keep the input k-space untouched
        recon_kspace[subsampled_mask == 1] = subsampled_slice[
            subsampled_mask == 1
        ].transpose(1, 2)
        # Compute the image for the reconstructed k-space
        recon_img = IFFT(recon_kspace)

        # Merge all the coils, each contribution being modulated by the CSM
        fused_recon_img = (recon_img * torch.conj(csm)).sum(axis=-1)

        # Plot the reconstruction with the contributions of all the coils
        scale_factor = fused_recon_img.abs().max()
        img = fused_recon_img.abs() / scale_factor
        img = img.cpu()  #  (Nrows, Ncols, Nframes)

        # frame_idx = 0
        # h = plt.figure()
        # plt.imshow(img[:, :, frame_idx], cmap="gray")
        # plt.savefig(
        #     str(results_dir / f"slice_{slice_idx}_frame_{frame_idx}.png"),
        #     bbox_inches="tight",
        # )
        # plt.close(h)

        # Plot the combined coils for every frame
        # This allows to make a video
        for frame_idx in range(img.shape[2]):
            h = plt.figure()
            plt.imshow(img[:, :, frame_idx], cmap="gray")
            plt.savefig(
                str(results_dir / f"slice_{slice_idx}_{frame_idx:04d}.png"),
                bbox_inches="tight",
            )
            plt.close(h)

        # Plot, for evey coil, the predicted image
        # for coil_idx in range(recon_img.shape[3]):
        #     img = (recon_img[:, :, frame_idx, coil_idx].abs() / scale_factor).cpu()
        #     h = plt.figure()
        #     plt.imshow(img, cmap="gray", clim=[0, 1])
        #     plt.savefig(
        #         str(
        #             results_dir
        #             / f"slice_{slice_idx}_frame_{frame_idx}_coil_{coil_idx}.png"
        #         ),
        #         bbox_inches="tight",
        #     )
        #     plt.close(h)
    return pre_intensity, csm, img


def train(rootdir, acc_factor, view, training_cfg):
    dataset = MICCAI2023(
        rootdir=rootdir,
        view=view,
        acc_factor=acc_factor,
    )

    # Take a random sample
    sample_idx = random.randint(0, len(dataset) - 1)
    # sample_idx = 0

    # Prepare the directory in which to store the results
    patient_path = dataset.patients[sample_idx]
    results_dir = pathlib.Path("./results") / patient_path.name

    # Prepare the results directory, rm if already exists
    if results_dir.exists():
        logging.info(f"Removing {results_dir} as it already exists")
        shutil.rmtree(results_dir)

    logging.info(f"Results will be saved into {results_dir}")
    results_dir.mkdir(parents=True)

    # Get the sample
    subsampled_data, subsampled_mask, fullsampled_data = dataset[sample_idx]

    # Subsampled_data and fullsampled_data are (kx, ky, sc, sz, t)
    n_coils = subsampled_data.shape[-3]
    n_slices = subsampled_data.shape[-2]
    n_frames = subsampled_data.shape[-1]

    # Iterate over the slices
    all_psnrs = []
    all_psnrs_sk = []
    all_ssim = []

    logging.info(f"Processing {n_slices} slices")
    for slice_idx in tqdm.tqdm(range(n_slices)):

        # Get the slices from the subsampled and fullsampled data
        # These slices are (kx, ky, sc, t)
        subsampled_slice = subsampled_data[:, :, :, slice_idx, :]
        fullsampled_slice = fullsampled_data[:, :, :, slice_idx, :]

        # Compute the normalization factor by computing the max RSS
        # of the images
        # This step is super important for the training to work properly
        coils_combined = combine_coils(subsampled_slice)
        norm_factor = coils_combined.max()
        logging.debug(
            f"For slice {slice_idx}, using the normalization factor {norm_factor}"
        )

        subsampled_slice = subsampled_slice / norm_factor.item()

        # Perform inference on this slice
        pre_intensity, csm, pred_image = infer_on_slice(
            subsampled_slice, subsampled_mask, slice_idx, results_dir, training_cfg
        )

        # Compute the PSNR for every slice and every coil
        # and also plot, for comparison,
        # - the full sampled image
        # - the zero filled image (the unobserved k-space is filled with zeros)
        # - the predicted image

        full_image = combine_coils(fullsampled_slice)  # nrows, ncols, nframes
        full_image /= full_image.max()

        zero_filled = combine_coils(subsampled_slice)
        zero_filled /= zero_filled.max()

        for frame_idx in range(full_image.shape[2]):
            img = full_image[:, :, frame_idx]
            zero_img = zero_filled[:, :, frame_idx]
            pred_img = pred_image[:, :, frame_idx]

            # Compute the PSNR
            mse = ((img - pred_img) ** 2).mean()
            data_range = 1.0
            psnr = 10.0 * np.log10(data_range**2 / mse)
            all_psnrs.append(psnr)

            psnr_sk = peak_signal_noise_ratio(img, pred_img, data_range=1)
            all_psnrs_sk.append(psnr_sk)

            ssim = structural_similarity(img, pred_img, data_range=1)
            all_ssim.append(ssim)

            logging.debug(
                f"For slice {slice_idx}, frame {frame_idx}: PSNR ({psnr}), PSNR (skimage) ({psnr_sk}), SSIM ({ssim})"
            )

            fig, axes = plt.subplots(nrows=1, ncols=3)
            axes[0].imshow(img, cmap="gray", clim=[0, 1])
            axes[0].set_title("Ground truth")
            axes[0].axis("off")

            axes[1].imshow(pred_img, cmap="gray", clim=[0, 1])
            axes[1].set_title("Predicted")
            axes[1].axis("off")

            axes[2].imshow(zero_img, cmap="gray", clim=[0, 1])
            axes[2].set_title("Zero filled")
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig(
                results_dir / f"slice_{slice_idx}_frame_{frame_idx}.png",
                bbox_inches="tight",
                dpi=150,
            )
            plt.close(fig)

    logging.info(
        f"Mean PSNR evaluated over all the slices and all the frames : {np.mean(all_psnrs)}"
    )
    logging.info(
        f"Mean PSNR (skimage) evaluated over all the slices and all the frames : {np.mean(all_psnrs_sk)}"
    )
    logging.info(
        f"Mean SSIM evaluated over all the slices and all the frames : {np.mean(all_ssim)}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Implementation of the neural implict neural representation for cine MRI reconstruction"
    )
    parser.add_argument(
        "--rootdir",
        type=str,
        default="data",
        help="Path to the data directory",
        required=True,
    )
    parser.add_argument(
        "--acc_factor",
        default=AccFactor.ACC4,
        type=AccFactor.__getitem__,
        help="Acceleration factor (ACC4, ACC8, ACC10)",
    )
    parser.add_argument(
        "--view",
        default=CINEView.SAX,
        type=CINEView.__getitem__,
        help="View of the cine MRI data (SAX, LAX)",
    )

    # Some training parameters
    parser.add_argument(
        "--lr",
        default=0.01,
        type=float,
        help="The base learning rate for the optimizer",
    )
    parser.add_argument(
        "--iter",
        default=512,
        type=int,
        help="The number of training iterations per slice",
    )
    parser.add_argument(
        "--reg_weight",
        default=4.0,
        type=float,
        help="The weights of the TV loss in the total loss",
    )

    args = parser.parse_args()

    training_cfg = {"lr": args.lr, "iter": args.iter, "reg_weight": args.reg_weight}

    train(args.rootdir, args.acc_factor, args.view, training_cfg)
