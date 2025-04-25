# External imports
import torch
from torch.optim.lr_scheduler import (
    StepLR,
    ReduceLROnPlateau,
    CyclicLR,
    OneCycleLR,
    CosineAnnealingLR,
)


def get_optimizer(cfg, params):
    params_dict = cfg["params"]
    exec(f"global optim; optim = torch.optim.{cfg['algo']}(params, **params_dict)")
    return optim


def get_scheduler(config, optimizer, steps_per_epoch):
    scheduler_name = config["scheduler"]["algo"]
    if scheduler_name == "OneCycleLR":
        config["scheduler"]["params"]["steps_per_epoch"] = steps_per_epoch
        config["scheduler"]["params"]["epochs"] = config["nepochs"]
        return OneCycleLR(optimizer, **config["scheduler"]["params"])
    elif scheduler_name == "CosineAnnealingLR":
        config["scheduler"]["params"]["T_max"] = config["nepochs"]
        return CosineAnnealingLR(optimizer, **config["scheduler"]["params"])
    else:
        return eval(
            f"torch.optim.lr_scheduler.{scheduler_name}(optimizer, **config['scheduler']['params'])"
        )
