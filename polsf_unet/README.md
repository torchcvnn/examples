# Semantic segmentation of PolSF with a complex valued UNet

This complete example provides a strong codebase for training a complex valued
UNet for the semantic segmentation of PolSF.

The codebase is parametrized by a yaml file in which you can customize your
experiment.

To use it :

```bash
python3 -m venv venv
source venv/bin/activate

python -m pip install -e .
<OR>
python -m pip install .
```

And then you can run the code by providing your `config.yaml` :

```
python -m torchtmpl.main train configs/config.yml
```
