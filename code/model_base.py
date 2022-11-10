import torch
import torch.nn as nn


def conv_block(in_channels, out_channels, dilation):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=dilation, stride=1, bias=False, dilation=dilation),
        nn.BatchNorm2d(out_channels, affine=True),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):

    def __init__(self, width, dilation):
        super().__init__()

        self.dconv_down1 = conv_block(1, width, dilation)
        self.dconv_down2 = conv_block(width, width, dilation)
        self.dconv_down3 = conv_block(width, 2*width, dilation)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up2 = conv_block(2*width + width, width, dilation)
        self.dconv_up1 = conv_block(width + width, width, dilation)

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