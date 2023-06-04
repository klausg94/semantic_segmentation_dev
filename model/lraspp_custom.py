# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 09:30:18 2022

@author: Z006AIKC
"""

from collections import OrderedDict
from typing import Any, Dict, Optional
import time

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import nn, Tensor
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation import LRASPP
from torchvision.utils import _log_api_usage_once


__all__ = ["lraspp_mobilenet_v3_custom"]


class LRASPPHead(nn.Module):
    def __init__(self, low_channels: int, high_channels: int, num_classes: int, inter_channels: int) -> None:
        super().__init__()
        
        if low_channels != -1:
            self.low_classifier = nn.Conv2d(low_channels, num_classes, 1)
        if high_channels != -1:
            self.cbr = nn.Sequential(
                nn.Conv2d(high_channels, inter_channels, 1, bias=False),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
            )
            self.scale = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(high_channels, inter_channels, 1, bias=False),
                nn.Sigmoid(),
            )
            self.high_classifier = nn.Conv2d(inter_channels, num_classes, 1)
            
    def forward(self, input: Dict[str, Tensor]) -> Tensor:
        l = 0
        h = 0
        if "low" in input.keys():
            low = input["low"]
            l = 1
        if "high" in input.keys():
            high = input["high"]
            x = self.cbr(high)
            s = self.scale(high)
            x = x * s
            if l == 1:
                x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
            h = 1
        if l==1 and h==1:
            return self.low_classifier(low) + self.high_classifier(x)
        if l==1 and h==0:
            return self.low_classifier(low)
        if l==0 and h==1:
            return self.high_classifier(x)
        

class LRASPP(nn.Module):
    """
    Implements a Lite R-ASPP Network for semantic segmentation from
    `"Searching for MobileNetV3"
    <https://arxiv.org/abs/1905.02244>`_.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "high" for the high level feature map and "low" for the low level feature map.
        low_channels (int): the number of channels of the low level features.
        high_channels (int): the number of channels of the high level features.
        num_classes (int, optional): number of output classes of the model (including the background).
        inter_channels (int, optional): the number of channels for intermediate computations.
    """

    def __init__(
        self, backbone: nn.Module, low_channels: int, high_channels: int, num_classes: int, inter_channels: int = 40,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.backbone = backbone
        self.classifier = LRASPPHead(low_channels, high_channels, num_classes, inter_channels)

    def forward(self, input: Tensor) -> Dict[str, Tensor]:
        features = self.backbone(input)
        out = self.classifier(features)
        out = F.interpolate(out, size=input.shape[-2:], mode="bilinear", align_corners=False)

        result = OrderedDict()
        result["out"] = out

        return result


def _lraspp_mobilenetv3(backbone, num_classes: int, low_index=-4, high_index=-1):
    backbone = backbone.features
    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    
    return_layer_dict = {}
    if low_index is not None:
        low_pos = stage_indices[low_index]  # use C2 here which has output_stride = 8
        low_channels = backbone[low_pos].out_channels
        return_layer_dict[str(low_pos)] = "low"
    else:
        low_channels = -1
    if high_index is not None:
        high_pos = stage_indices[high_index]  # use C5 which has output_stride = 16
        high_channels = backbone[high_pos].out_channels
        return_layer_dict[str(high_pos)] = "high"
    else:
        high_channels = -1
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layer_dict)
    return LRASPP(backbone, low_channels, high_channels, num_classes, max(30, int(1.*high_channels/7.)))


def lraspp_mobilenet_v3_custom(mobilenetv3version = "small", num_classes = 3, low_index=-4, high_index=-1, dilated=True):
    if mobilenetv3version == "small":
        """
        Intermediate Layer Info for input size (1000,1000)
        STAGE INDICES:  [0, 1, 2, 4, 9, 12] 

        torch.Size([1, 16, 500, 500])
        torch.Size([1, 16, 250, 250])
        torch.Size([1, 24, 125, 125])
        torch.Size([1, 40, 63, 63])
        torch.Size([1, 96, 63, 63])
        torch.Size([1, 576, 63, 63])
        """
        backbone = torchvision.models.mobilenet_v3_small(weights="MobileNet_V3_Small_Weights.IMAGENET1K_V1", dilated=dilated)
    elif mobilenetv3version == "large_imagenet":
        """
        Intermediate Layer Info for input size (1000,1000)
        STAGE INDICES:  [0, 2, 4, 7, 13, 16]
        DEFAULT LRASPP: low_index=-4, high_index=-1

        torch.Size([1, 16, 500, 500])
        torch.Size([1, 24, 250, 250])
        torch.Size([1, 40, 125, 125])
        torch.Size([1, 80, 63, 63])
        torch.Size([1, 160, 63, 63])
        torch.Size([1, 960, 63, 63])
        """
        backbone = torchvision.models.mobilenet_v3_large(weights="MobileNet_V3_Large_Weights.IMAGENET1K_V1", dilated=dilated)
    elif mobilenetv3version == "large_coco":
        model_ = torchvision.models.segmentation.lraspp_mobilenet_v3_large(weights='COCO_WITH_VOC_LABELS_V1')
        backbone = torchvision.models.mobilenet_v3_large(weights="MobileNet_V3_Large_Weights.IMAGENET1K_V1", dilated=True)
        backbone.features.load_state_dict(model_.backbone.state_dict())
    else:
        raise Exception()
    return _lraspp_mobilenetv3(backbone, num_classes, low_index=low_index, high_index=high_index)