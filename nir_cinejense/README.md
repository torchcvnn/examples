# Neural Implicit Representation for cardiac reconstruction

This example reproduces the code from [https://github.com/MDL-UzL/CineJENSE](https://github.com/MDL-UzL/CineJENSE) along the paper "CineJENSE: Simultaneous Cine MRI Image Reconstruction and Sensitivity Map Estimation using Neural Representations" by Ziad Al-Haj Hemidi, Nora Vogt, Lucile Quillien, Christian Weihsbach, Mattias P. Heinrich, and Julien Oster.

It is showcasing NIR for cardiac reconstruction. In a few words : MRI is
sampling in the Fourier space as bands. The longer the exam, the more you
collect bands from the Fourier representation. The task is to estimate the
non-observed bands from the observed bands, hopefully to get an exam as short as
possible but still observing the heart as if it was observed for a longer
period.

For running this example, you need to download the data from the [CMRxRecon MICCAI 2023
challenge](https://cmrxrecon.github.io/Home.html). See also the github of the
challenge to access the data [https://github.com/CmrxRecon/CMRxRecon2024](https://github.com/CmrxRecon/CMRxRecon2024)

The data directory is expected to follow the structure used by [torchcvnn](https://torchcvnn.github.io/torchcvnn/modules/datasets.html#torchcvnn.datasets.MICCAI2023).

	rootdir/ChallengeData/MultiCoil/cine/TrainingSet/P{id}/
								- cine_sax.mat
								- cin_lax.mat
	rootdir/ChallengeData/MultiCoil/cine/TrainingSet/AccFactor04/P{id}/
								- cine_sax.mat
								- cine_sax_mask.mat
								- cin_lax.mat
								- cine_lax_mask.mat


The script supports the three acceleration factors and both the Short Axis (SAX) and
Long Axis (LAX).

If multiple patient data are provided, the script will sample one of them randomly.

The installation is done in two steps 

```bash
python -m pip install -r requirements.txt
python -m pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

The tinycudann installation may complain if the version of torch, installed during the first step is not using the cuda version for which you have installed the librairies. You may need to overwrite the installed torch version by installing the one supporting the right cuda version and listed on [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions).

Once the dependencies are installed and you have the data available, you should be able to run the code : 

```
python nir_miccai2023.py --rootdir /path/to/the/data --acc_factor ACC10 --view SAX
```

## Examples

The examples below have been produced on a GTX GeForce 3090, taking 2 minutes per slice.

They were executed with numpy==1.26.4 tinycudann==1.7 torchcvnn==0.8.0 torch==2.0.1

- ACC4, SAX, patient P002, mean PSNR=$42.19$ (mean over the $12$ frames, for slice number $5$)

![ACC4, SAX, P002](https://github.com/torchcvnn/examples/blob/nir_cinejense/nir_cinejense/gifs/acc4_sax_p002.gif?raw=true)


- ACC10, LAX, patient P014, PSNR=

![ACC10, LAX, P014](https://github.com/torchcvnn/examples/blob/nir_cinejense/nir_cinejense/gifs/acc10_lax_p014.gif?raw=true)


