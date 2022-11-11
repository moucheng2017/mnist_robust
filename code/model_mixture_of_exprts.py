import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_channels, out_channels, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=dilation, stride=1, bias=False, dilation=dilation),
        nn.BatchNorm2d(out_channels, affine=True),
        nn.ReLU(inplace=True)
    )


class UNetMoE(nn.Module):

    def __init__(self, width, dilation=5):
        super().__init__()

        self.dconv_down1a = conv_block(1, width, 1)
        self.dconv_down2a = conv_block(width, width, 1)
        self.dconv_down3a = conv_block(width, 2*width, 1)

        self.dconv_down1b = conv_block(1, width, dilation)
        self.dconv_down2b = conv_block(width, width, dilation)
        self.dconv_down3b = conv_block(width, 2*width, dilation)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up2a = conv_block(2*width + width, width, 1)
        self.dconv_up1a = conv_block(width + width, width, 1)

        self.dconv_up2b = conv_block(2*width + width, width, dilation)
        self.dconv_up1b = conv_block(width + width, width, dilation)

        self.conv_last = nn.Conv2d(width, 1, 1)

    def forward(self, x):

        conv1a = self.dconv_down1a(x)
        conv1b = self.dconv_down1b(x)
        conv1 = conv1a + conv1b
        x = self.maxpool(conv1)

        conv2a = self.dconv_down2a(x)
        conv2b = self.dconv_down2b(x)
        conv2 = conv2a + conv2b
        x = self.maxpool(conv2)

        conv3a = self.dconv_down3a(x)
        conv3b = self.dconv_down3b(x)
        conv3 = conv3a + conv3b
        x = self.upsample(conv3)

        x = torch.cat([x, conv2], dim=1)
        xa = self.dconv_up2a(x)
        xb = self.dconv_up2b(x)
        x = xa+xb
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        xa = self.dconv_up1a(x)
        xb = self.dconv_up1b(x)
        x = xa+xb
        out = self.conv_last(x)

        return out