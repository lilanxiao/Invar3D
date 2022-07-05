import torch
import torch.nn as nn
import torchvision as tv

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from normconv import NormDepthConv2d
from typing import List

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, se_module=False):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        
        self.se_module = se_module
        if se_module:
            self.fc1 = nn.Linear(out_channels, out_channels)
            self.fc2 = nn.Linear(out_channels, out_channels)
            
            self.conv1 = nn.Conv2d(out_channels, out_channels, 1)
            self.conv2 = nn.Conv2d(out_channels, 1, 1)
            self.bn1 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        if not self.se_module:
            return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
        else:
            residual = self.residual_function(x)
            # spectral attention
            y = nn.functional.adaptive_avg_pool2d(residual, output_size=1)
            y = y.squeeze(-1).squeeze(-1)
            y = self.fc1(y)
            y = nn.functional.relu(y, True)
            y = self.fc2(y)
            y = torch.sigmoid(y).unsqueeze(-1).unsqueeze(-1)
            y = y * residual

            # spatial attention
            z = nn.functional.relu(self.bn1(self.conv1(y)), True)
            z = self.conv2(z)
            z = torch.sigmoid(z)
            y = y * z
            return nn.functional.relu(y + self.shortcut(x), True)

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )
        
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
    
class ResNetFusion(nn.Module):

    def __init__(self, block, num_block, dims_in=1, ret_inter=False):
        super().__init__()

        self.in_channels = 64
        self.ret_inter = ret_inter

        self.conv1 = nn.Sequential(
            NormDepthConv2d(output_dim=32, padding=3, kernel_size=7, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True))

        self.conv2 = nn.Sequential(
             nn.Conv2d(1, 32, 7, padding=3, stride=2, bias=False),
             nn.BatchNorm2d(32),
             nn.ReLU(True))

        self.conv_rgb = nn.Sequential(
             nn.Conv2d(3, 32, 7, padding=3, stride=2, bias=False),
             nn.BatchNorm2d(32),
             nn.ReLU(True))

        self.conv2_x = self._make_layer(block, 64, num_block[0], 2)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 256, num_block[3], 2)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        
        return nn.Sequential(*layers)

    def forward(self, endpoint):
        x = endpoint["depthmap"]
        rgb = endpoint["rgb"]

        output = self.conv1(x) + self.conv2(x)
        y = self.conv_rgb(rgb)
        output = torch.cat([output, y], dim=1)

        x1 = output
        output = self.conv2_x(output)
        x2 = output
        output = self.conv3_x(output)
        x3 = output
        output = self.conv4_x(output)
        x4 = output
        output = self.conv5_x(output)
        if self.ret_inter:
            return x1, x2, x3, x4, output
        else:
            return output 


class ResNet(nn.Module):

    def __init__(self, block, num_block, dims_in=1, ret_inter=False, rgb = False):
        super().__init__()

        self.in_channels = 64
        self.ret_inter = ret_inter
        self.rgb = rgb
        if rgb:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, 7, padding=3, stride=2, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True))
        else:
            self.conv1 = nn.Sequential(
                # nn.Conv2d(dims_in, 64, kernel_size=7, padding=3, stride=2, bias=False),
                NormDepthConv2d(padding=3, kernel_size=7, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(True))

            self.conv2 = nn.Sequential(
                nn.Conv2d(1, 64, 7, padding=3, stride=2, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True))      

        self.conv2_x = self._make_layer(block, 64, num_block[0], 2)     # strided conv2d instead of pooling
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 256, num_block[3], 2)
        # self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        # we have num_block blocks per layer, the first block 
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        
        return nn.Sequential(*layers)

    def forward(self, endpoint):
        # x = endpoint["depthmap"]
        x = endpoint

        if self.rgb:
            output = self.conv1(x)
        else:
            output = self.conv1(x) + self.conv2(x)

        x1 = output
        output = self.conv2_x(output)
        x2 = output
        output = self.conv3_x(output)
        x3 = output
        output = self.conv4_x(output)
        x4 = output
        output = self.conv5_x(output)
        if self.ret_inter:
            return x1, x2, x3, x4, output
        else:
            return output 

def resnet18(ret_inter=False):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], ret_inter=ret_inter)

def resnet34(ret_inter=False, rgb=False):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], ret_inter=ret_inter, rgb=rgb)

def resnet50(block=BottleNeck, ret_inter=False):
    """ return a ResNet 50 object
    """
    return ResNet(block, [3, 4, 6, 3], ret_inter=ret_inter)

def resnet101(block=BottleNeck, ret_inter=False):
    """ return a ResNet 101 object
    """
    return ResNet(block, [3, 4, 23, 3], ret_inter=ret_inter)

def resnet152(block=BottleNeck, ret_inter=False):
    """ return a ResNet 152 object
    """
    return ResNet(block, [3, 8, 36, 3], ret_inter=ret_inter)

class UNet(nn.Module):
    def __init__(self, bottle_neck=False, rgb=False, pretrain=False, fusion=False):
        super().__init__()
        if bottle_neck:
            self.resnet = resnet50(ret_inter=True)
        else:
            self.resnet = resnet34(True, rgb)

        self.fusion = fusion
        if fusion:
            print("using fused model ")
            self.resnet = ResNetFusion(BasicBlock, [3, 4, 6, 3], ret_inter=True)
        
        # use pretrained ResNet
        if pretrain and rgb:
            print("using pretrained ResNet34...")
            self.resnet = ResNet34Pretrained(256)

        ef = 3*int(bottle_neck) + 1     # expansion factor
        self.upsample1 = nn.Sequential(nn.ConvTranspose2d(256 * ef, 256 *ef, kernel_size=2, stride=2, padding=0, bias=False),
                                        nn.BatchNorm2d(256 * ef),
                                        nn.LeakyReLU(),
                                        BasicBlock(256 * ef, 256 * ef))
        self.upsample2 = nn.Sequential(nn.ConvTranspose2d(512 * ef, 256 *ef, kernel_size=2, stride=2, padding=0, bias=False),
                                        nn.BatchNorm2d(256 * ef),
                                        nn.LeakyReLU(),
                                        BasicBlock(256 * ef, 256 * ef))
        if bottle_neck:
            self.conv = nn.Sequential(nn.Conv2d((256 + 128) * ef, 256 + 128, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(256 + 128),
                                        nn.ReLU(True),
                                        BasicBlock(256+128, 256))
        else:
            self.conv = BasicBlock(256+128, 256)

    def forward(self, x, return_feature_map = False):
        x1, x2, x3, x4, x5 = self.resnet(x)
        y1 = self.upsample1(x5)
        y2 = self.upsample2(torch.cat([x4, y1], dim=1))
        y3 = self.conv(torch.cat([x3, y2], dim=1))
        if return_feature_map:
            return y3, x1, x2
        else:
            return y3
        

class ResNet34Pretrained(torch.nn.Module):
    def __init__(self, out_dims=256) -> None:
        super().__init__()
        net = tv.models.resnet.resnet34(pretrained=True)
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        
        self.last = nn.Sequential(
            nn.Conv2d(512, out_dims, 1, bias=False),
            nn.BatchNorm2d(out_dims),
            nn.ReLU(True)
        )
    def forward(self, x):
        x = self.conv1(x)
        x1 = self.relu(self.bn1(x))
        x2 = self.layer1(self.maxpool(x1))
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.last(self.layer4(x4))
        return x1, x2, x3, x4, x5
    

from typing import Dict
from torch import Tensor
class UNetCAM(nn.Module):
    """with support for CAM Conv"""
    def __init__(self, dim_cam_feat, bottle_neck=False, rgb=False, pretrain=False, fusion=False):
        super().__init__()
        if bottle_neck:
            self.resnet = resnet50(ret_inter=True)
        else:
            self.resnet = resnet34(True, rgb)

        self.fusion = fusion
        if fusion:
            print("using fused model ")
            self.resnet = ResNetFusion(BasicBlock, [3, 4, 6, 3], ret_inter=True)
        
        # use pretrained ResNet
        if pretrain and rgb:
            print("using pretrained ResNet34...")
            self.resnet = ResNet34Pretrained(256)

        ef = 3*int(bottle_neck) + 1     # expansion factor
        self.upsample1 = nn.Sequential(nn.ConvTranspose2d(256 * ef, 256 *ef, kernel_size=2, stride=2, padding=0, bias=False),
                                        nn.BatchNorm2d(256 * ef),
                                        nn.LeakyReLU(),
                                        BasicBlock(256 * ef, 256 * ef))
        self.upsample2 = nn.Sequential(nn.ConvTranspose2d(512 * ef, 256 *ef, kernel_size=2, stride=2, padding=0, bias=False),
                                        nn.BatchNorm2d(256 * ef),
                                        nn.LeakyReLU(),
                                        BasicBlock(256 * ef, 256 * ef))
        if bottle_neck:
            self.conv = nn.Sequential(nn.Conv2d((256 + 128) * ef, 256 + 128, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(256 + 128),
                                        nn.ReLU(True),
                                        BasicBlock(256+128, 256))
        else:
            self.conv = BasicBlock(256+128, 256)
        
        self.dim_cam_feat = dim_cam_feat
        self.camconv1 = nn.Sequential(
            nn.Conv2d(256 + dim_cam_feat, 256, 3, bias=False, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        self.camconv2 = nn.Sequential(
            nn.Conv2d(256 + dim_cam_feat, 256, 3, bias=False, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        self.camconv3 = nn.Sequential(
            nn.Conv2d(128 + dim_cam_feat, 128, 3, bias=False, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )

    def forward(self, x:Dict[str, Tensor]):
        depth = x["depthmap"]
        fs8 = x["cam_feat_s8"]
        fs16 = x["cam_feat_s16"]
        fs32 = x["cam_feat_s32"]
        x1, x2, x3, x4, x5 = self.resnet(depth)
        # augument feature maps with camera intrinsic
        x5 = self.camconv1(torch.cat([x5, fs32], dim=1))
        x4 = self.camconv2(torch.cat([x4, fs16], dim=1))
        x3 = self.camconv3(torch.cat([x3, fs8], dim=1))
        # upsample 
        y1 = self.upsample1(x5)
        y2 = self.upsample2(torch.cat([x4, y1], dim=1))
        y3 = self.conv(torch.cat([x3, y2], dim=1))
        return y3
    