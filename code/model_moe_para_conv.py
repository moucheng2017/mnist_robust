import torch
import torch.nn as nn
import torch.nn.functional as F
from parallel_classes import conv2D_para


class UNetMoE_flexi(nn.Module):

    def __init__(self, width, dilation=1):
        super().__init__()

        self.dconv_down1 = conv2D_para(1, width, 1, num_experts=2)  # conv_block(1, width, 1)
        self.dconv_down2 = conv2D_para(width, width, 1, num_experts=2)
        self.dconv_down3 = conv2D_para(width, 2*width, 1, num_experts=2)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up2 = conv2D_para(2*width + width, width, 1, num_experts=2)
        self.dconv_up1 = conv2D_para(width + width, width, 1, num_experts=2)

        self.conv_last = nn.Conv2d(width, 1, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.upsample(conv3)

        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)

        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out