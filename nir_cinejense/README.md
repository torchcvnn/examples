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

```bash
python -m pip install -r requirements.txt
python nir_miccai2023.py --rootdir /path/to/the/data --acc_factor ACC10 --view SAX
```

Below are shown some examples for different acceleration factors and LAX/SAX.
