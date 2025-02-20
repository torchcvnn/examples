# External imports
import torch
from torch.nn import MSELoss, L1Loss, SmoothL1Loss, CrossEntropyLoss, NLLLoss
from torchcvnn.nn.modules.loss import ComplexMSELoss
from torchtmpl.losses import (
    ComplexCrossEntropyLoss,
    FocalLoss,
)
from torch.optim.lr_scheduler import (
    StepLR,
    ReduceLROnPlateau,
    CyclicLR,
    OneCycleLR,
    CosineAnnealingLR,
)


def get_loss(config, class_weights=None, ignore_index=-100, gamma=0.0):
    loss_name = config["loss"]["name"]
    if loss_name == "ComplexCrossEntropyLoss" and class_weights is not None:
        return ComplexCrossEntropyLoss(
            class_weights=class_weights, ignore_index=ignore_index
        )
    elif loss_name == "FocalLoss" and class_weights is not None:
        gamma = config["loss"]["gamma"]
        return FocalLoss(alpha=class_weights, ignore_index=ignore_index, gamma=gamma)
    else:
        return globals()[loss_name]()


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
