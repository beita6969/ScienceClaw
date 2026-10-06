"""
 ERFNet full model definition for Pytorch
 Sept 2017
 Eduardo Romera

 Attribution to: https://github.com/Eromera/erfnet_pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput, batch_norm=False, instance_norm=False):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput,
                              (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.batch_norm = batch_norm
        self.instance_norm = instance_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        if self.instance_norm:
            self.in_ = torch.nn.InstanceNorm2d(noutput)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        if self.batch_norm:
            output = self.bn(output)
        if self.instance_norm:
            output = self.in_(output)
        return F.relu(output)


class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated, batch_norm=False, instance_norm=False):
        super().__init__()
        self.batch_norm = batch_norm
        self.instance_norm = instance_norm

        self.conv3x1_1 = nn.Conv2d(
            chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(
            chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)
        if self.instance_norm:
            self.in1_ = torch.nn.InstanceNorm2d(chann)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(
            1*dilated, 0), bias=True, dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(
            0, 1*dilated), bias=True, dilation=(1, dilated))

        if self.batch_norm:
            self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)
        if self.instance_norm:
            self.in2_ = torch.nn.InstanceNorm2d(chann)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        if self.batch_norm:
            output = self.bn1(output)
        if self.instance_norm:
            output = self.in1_(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        if self.batch_norm:
            output = self.bn2(output)
        if self.instance_norm:
            output = self.in2_(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output+input)  # +input = identity (residual connection)


class Encoder(nn.Module):
    def __init__(self, num_classes, batch_norm=False, instance_norm=False):
        super().__init__()

        self.initial_block = DownsamplerBlock(3, 16, batch_norm, instance_norm)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64, batch_norm, instance_norm))

        DROPOUT = 0.0 #0.03
        for x in range(0, 5):  # 5 times
            self.layers.append(non_bottleneck_1d(64, DROPOUT, 1, batch_norm, instance_norm))

        self.layers.append(DownsamplerBlock(64, 128, batch_norm, instance_norm))


        DROPOUT = 0.0 #0.3
        for x in range(0, 2):  # 2 times
            self.layers.append(non_bottleneck_1d(128, DROPOUT, 2, batch_norm, instance_norm))
            self.layers.append(non_bottleneck_1d(128, DROPOUT, 4, batch_norm, instance_norm))
            self.layers.append(non_bottleneck_1d(128, DROPOUT, 8, batch_norm, instance_norm))
            self.layers.append(non_bottleneck_1d(128, DROPOUT, 16, batch_norm, instance_norm))

        # Only in encoder mode:
        self.output_conv = nn.Conv2d(
            128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)
        for layer in self.layers:
            output = layer(output)

        if predict:
            output = self.output_conv(output)

        return output


class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput, batch_norm=False, instance_norm=False):
        super().__init__()
        self.batch_norm = batch_norm
        self.instance_norm = instance_norm

        self.conv = nn.ConvTranspose2d(
            ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        if self.instance_norm:
            self.in_ = torch.nn.InstanceNorm2d(noutput)

    def forward(self, input):
        output = self.conv(input)
        if self.batch_norm:
            output = self.bn(output)
        if self.instance_norm:
            output = self.in_(output)
        return F.relu(output)


class Decoder (nn.Module):
    def __init__(self, num_classes, batch_norm=False, instance_norm=False):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128, 64, batch_norm, instance_norm))
        self.layers.append(non_bottleneck_1d(64, 0, 1, batch_norm, instance_norm))
        self.layers.append(non_bottleneck_1d(64, 0, 1, batch_norm, instance_norm))

        self.layers.append(UpsamplerBlock(64, 16, batch_norm, instance_norm))
        self.layers.append(non_bottleneck_1d(16, 0, 1, batch_norm, instance_norm))
        self.layers.append(non_bottleneck_1d(16, 0, 1, batch_norm, instance_norm))

        self.output_conv = nn.ConvTranspose2d(
            16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output

# ERFNet


class Net(nn.Module):
    def __init__(self, num_classes, encoder=None, batch_norm=False, instance_norm=False):  # use encoder to pass pretrained encoder
        super().__init__()

        if (encoder == None):
            self.encoder = Encoder(num_classes, batch_norm, instance_norm)
        else:
            self.encoder = encoder
        self.decoder = Decoder(num_classes, batch_norm, instance_norm)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)  # predict=False by default
            return self.decoder.forward(output)
