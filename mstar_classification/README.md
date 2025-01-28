# MSTAR classification with patched CNNs and complex-valued ViTs

This example demonstrates how to patch a real-valued neural network to replace the real-valued modules by complex-valued modules as well as how to define a complex-valued ViTs. These networks are trained to classify MSTAR.

```bash
python -m pip install torchcvnn
python -m pip install seaborn matplotlib torchvision torchmetrics lightning
python train.py --version 0 --datadir MSTAR_DATADIR 
```
