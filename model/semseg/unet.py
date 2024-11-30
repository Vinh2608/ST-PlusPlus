from __future__ import division, print_function

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
from model.semseg.base import BaseNet
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class UNet(BaseNet):
    def __init__(self, backbone, class_num, args):
        super(UNet, self).__init__(backbone)

        self.model = smp.Unet(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=3,
            classes=3,
            activation='softmax2d',
        )
        self.name = args.dataset

    def base_forward(self, x):
        output = self.model(x)
        return output
    
    def forward(self, x, tta=False):
        if not tta:
            return self.base_forward(x)

        else:
            h, w = x.shape[-2:]
            if self.name == 'dataset2':
                scales = [0.5, 0.8, 1.0, 1.5, 2.0]
            if self.name == 'lisc' or self.name == 'dataset1':
                scales = [0.5, 0.75, 1.0, 1.5, 2.0]

            final_result = None

            for scale in scales:
                cur_h, cur_w = int(h * scale), int(w * scale)
                cur_x = F.interpolate(x, size=(cur_h, cur_w), mode='bilinear', align_corners=True)

                out = F.softmax(self.base_forward(cur_x), dim=1)
                out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
                final_result = out if final_result is None else (final_result + out)

                out = F.softmax(self.base_forward(cur_x.flip(3)), dim=1).flip(3)
                out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
                final_result += out

            return final_result        