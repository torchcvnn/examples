# MSTAR classification with patched CNNs and complex-valued ViTs

This example demonstrates how to patch a real-valued neural network to replace the real-valued modules by complex-valued modules as well as how to define a complex-valued ViTs. These networks are trained to classify MSTAR.

```bash
python -m pip install torchcvnn
python -m pip install seaborn matplotlib torchvision torchmetrics lightning monai tensorboard
python train.py --version 0 --datadir MSTAR_DATADIR
```

## Dataset
The MSTAR [(Moving and Stationary Target Acquisition and Recognition)](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/3370/1/Moving-and-stationary-target-acquisition-and-recognition-MSTAR-model-based/10.1117/12.321851.short) dataset is a benchmark in SAR imaging and automatic target recognition. 

Samples per class

| Class   | A04 | A05 | A07 | A10 | A32 | A62 | A63 | A64  |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|
| Samples | 573 | 573 | 573 | 567 | 572 | 573 | 573 | 1417 |

| Class   | BTR-60 | 2S1 | BRDM_2 | D7  | SLICY | T62 | ZIL131 | ZSU_23_4 |
|---------|:------:|:---:|:------:|:---:|:-----:|:---:|:------:|:--------:|
| Samples | 573    | 573 | 573    | 567 | 572   | 573 | 573    | 1417     |

The dataset needs to be downloaded manually. Please visit [torchcvnn](https://torchcvnn.github.io/torchcvnn/modules/datasets.html#mstar) for detailed instructions.

## Model details
We currently support 3 architectures: ResNet-18, ViT and Hybrid-ViT. By default, hybrid-vit is set. You can change by parsing the argument "--model_type".

```bash
python train.py --version 0 --datadir MSTAR_DATADIR --model_type resnet18
```

Below are some examples of Vision Transformers architectures.
|   Models   | Layers | Heads | Hidden dimension |
|------------|:------:|:-----:|:----------------:|
| ViT_1      |    3   |   8   |       256        |
| ViT_2      |    3   |   8   |       128        |
| ViT_3      |    2   |   4   |       128        |
| Hybrid-ViT |    3   |   8   |       128        |

## Classification performance
| Models     | Params | Input size | Top-1 Accuracy | Top-5 Accuracy |
|------------|:------:|:----------:|:--------------:|:--------------:|
| ResNet-18  | 11.2M  | 128        | 99.8%          | 100%           |
| ViT_1      | 392K   | 128        | 56.8%          | 91.4%          |
| ViT_2      | 110K   | 128        | 81.2%          | 99.3%          |
| ViT_2      | 116K   | 208        | 89.8%          | 99.8%          |
| ViT_3      | 87.6K  | 208        | 85.7%          | 99.5%          |
| Hybrid-ViT | 695K   | 128        | 87.5%          | 99.7%          |
| Hybrid-ViT | 709K   | 208        | 91.1%          | 99.8%          |