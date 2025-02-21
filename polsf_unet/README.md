# Semantic segmentation of PolSF with a complex valued UNet

This complete example provides a strong codebase for training a complex valued
UNet for the semantic segmentation of PolSF.

The codebase is parametrized by a yaml file in which you can customize your
experiment.

## Setup

To use it :

```bash
python3 -m venv venv
source venv/bin/activate

python -m pip install -e .
<OR>
python -m pip install .
```

## Getting the data

You need to download manually the data. To download the PolSF dataset, please follow the following link: [https://ietr-lab.univ-rennes1.fr/polsarpro-bio/san-francisco/dataset/SAN_FRANCISCO_ALOS2.zip](https://ietr-lab.univ-rennes1.fr/polsarpro-bio/san-francisco/dataset/SAN_FRANCISCO_ALOS2.zip). **Important**: you must move the file SF-ALOS2-label2d.png from the sub-folder Config_labels to the main folder.

## Training

For training, you need to provide a YAML configuration file. As an example, you
can check the `configs/config.yml`. At least you need to adapt the path to the
data with the `data/dataset/trainpath` key.


And then you can run the code by providing your `config.yaml` :

```
python -m torchtmpl.main train configs/config.yml
```

With the config we provide, a low end GPU will be sufficient. It requires $3 GB$
of VRAM.

The best model will be saved into a directory specific to the run into the `logs` subdirectory.

## Testing

For testing the model on the test fold, it is as simple as running the library
with the log directory of your run

```
python -m torchtmpl.main test logs/UNet_0
```

This will reload the right configuration, the seed used during training (to
guarantee we use the same folds), the best model and evaluate it. The outputs
are saved in the run logdir.


