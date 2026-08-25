import torch
import torchvision
from typing import List, Callable
from agents.models.robomimic.models.obs_core import VisualCore


def get_resnet(input_shape: List[int], output_size: int, pretrained: bool = False):
    """Get ResNet model from torchvision.models
    Args:
        input_shape: Shape of input image (C, H, W).
        output_size: Size of output feature vector.
        pretrained: ImageNet initialisation of the ResNet-18 trunk.

    ── Gen14 U9 ────────────────────────────────────────────────────────────────────────
    `pretrained` is ADDITIVE: the default False reproduces every pre-U9 call site exactly
    (this file is vendored and shared by Gen6V4, Gen7, imf_visual_aligning,
    mix_visual_avoiding, ... — none of them pass the kwarg, so none of them change).

    🔴 Two things to know before trusting a `pretrained=True` run:
      1. MultiImageObsEncoder(use_group_norm=True) REPLACES every BatchNorm2d in this
         trunk with a freshly-initialised GroupNorm (multi_image_obs_encoder.py:62-69),
         discarding the pretrained affine params and running stats. The conv filters
         survive — which is where the transferable structure lives — but the network
         arrives DECALIBRATED. Prefer a reduced encoder LR (Trainer(vis_lr_scale=...))
         over a hard freeze.
      2. `pretrained=True` downloads into ~/.cache/torch/hub/checkpoints/ and compute
         nodes have no internet. Pre-fetch once on the login node. Gate G-B11 turns a
         silent fallback to random weights into a loud failure.
    ─────────────────────────────────────────────────────────────────────────────────────
    """

    resnet = VisualCore(
        input_shape=input_shape,
        backbone_class="ResNet18Conv",
        backbone_kwargs=dict(
            input_coord_conv=False,
            pretrained=bool(pretrained),
        ),
        pool_class="SpatialSoftmax",
        pool_kwargs=dict(
            num_kp=32,
            learnable_temperature=False,
            temperature=1.0,
            noise_std=0.0,
            output_variance=False,
        ),
        flatten=True,
        feature_dimension=output_size,
    )

    return resnet


def _get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    num_fc_in = resnet.fc.in_features

    resnet.fc = torch.nn.Linear(num_fc_in, 64)
    # resnet.fc = torch.nn.Identity()

    return resnet

def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m
    r3m.device = 'cpu'
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to('cpu')
    return resnet_model
