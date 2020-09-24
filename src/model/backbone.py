import math
import torch
import numpy as np
import torch.nn as nn

from collections import OrderedDict


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        
        super(SEModule, self).__init__()
        
        self.avg_pool = nn.AvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        module_input = x
        
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """

    def forward(self, x):
        
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        
        super(SEResNeXtBottleneck, self).__init__()
        
        width = int(math.floor(planes * (base_width / 64)) * groups)

        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):
    
    def __init__(self, block, layers, groups, reduction, dropout_p=0.2, in_channels=3,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000):
        """
        Squeeze-and-Excitation Networks
        ResNet code gently borrowed from
        https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

        Args:
            block (nn.Module): Bottleneck class.
            layers (list of ints): Number of residual blocks for 4 layers of the network.
            groups (int): Number of groups for the 3x3 convolution in each bottleneck block.
            reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            dropout_p (float or None): Drop probability for the Dropout layer.
            in_channels (int):  Number of input channels for layer 0.
            inplanes (int):  Number of input channels for layer1.
            input_3x3 (bool): If `True`, use three 3x3 convolutions instead of a single 7x7 
                              convolution in layer0.
            downsample_kernel_size (int): Kernel size for downsampling convolutions in 
                                          layer2, layer3 and layer4.
            downsample_padding (int): Padding for downsampling convolutions in layer2, layer3 
                                      and layer4.
            num_classes (int): Number of outputs in `last_linear` layer.
        """
        super(SENet, self).__init__()
        
        self.in_channels = in_channels
        self.inplanes = inplanes

        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(in_channels, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(in_channels, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]

        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2, ceil_mode=True)))
        
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout2d(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x):
        
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x

    def logits(self, x):
        
        x = self.avg_pool(x)
        
        if self.dropout is not None:
            x = dropout(x)
            
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        
        return x

    def forward(self, x):
        
        x = self.features(x)
        x = self.logits(x)
        
        return x


class DualAttention(nn.Module):

    """
    Dual Attention Network for Scene Segmentation
    https://arxiv.org/abs/1809.02983
    """

    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        
        super(DualAttention, self).__init__()
        
        inter_channels = in_channels // 4
        
        self.glob = nn.Sequential(nn.Conv2d(512 * 4, 512, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(512),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(inplace=True))

        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU(inplace=True))
        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU(inplace=True))

        self.sa = PositionAttention(inter_channels)
        self.sc = ChannelAttention(inter_channels)

        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 3, padding=1, bias=False),
                                    norm_layer(out_channels),
                                    nn.ReLU(inplace=True))
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 3, padding=1, bias=False),
                                    norm_layer(out_channels),
                                    nn.ReLU(inplace=True))

    def forward(self, x):
        
        x_glob = self.glob(x)

        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)

        feat_sum = x_glob + sa_conv + sc_conv
        
        return feat_sum


class PositionAttention(nn.Module):
    
    '''
    Position Attention Module
    Reference from SAGAN https://github.com/heykeetae/Self-Attention-GAN
    '''

    def __init__(self, in_dim):
        
        super(PositionAttention, self).__init__()
        
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps( B X C X H X W)
        returns :
            out : attention value + input feature
            attention: B X (HxW) X (HxW)
        """
        
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).contiguous().view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        
        return out


class ChannelAttention(nn.Module):
    '''
    Channel Attention Module
    '''

    def __init__(self, in_dim):
        
        super(ChannelAttention, self).__init__()
        
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        
        '''
        inputs :
            x : input feature maps( B X C X H X W)
        output :
            out : attention value + input feature
            attention: B X C X C
        '''
        
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)

        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        
        return out


class DA_SCSE_ResNeXt50(nn.Module):
    
    '''
    UNet backbone architecture based on ResNeXt50 
    + Squeeze Excitation block + Dual Attention
    '''

    
    def __init__(self, in_channels=3):
        
        super(DA_SCSE_ResNeXt50, self).__init__()

        self.encoder = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                             dropout_p=None, inplanes=64, input_3x3=False,
                             downsample_kernel_size=1, downsample_padding=0,
                             num_classes=1000, in_channels=3)

        self.relu = nn.ReLU(inplace=True)
        
        self.conv1 = nn.Sequential(self.encoder.layer0.conv1,
                                   self.encoder.layer0.bn1,
                                   self.encoder.layer0.relu1)

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center_global_pool = nn.AdaptiveAvgPool2d([1, 1])
        
        self.center_conv1x1 = nn.Conv2d(512 * 4, 64, kernel_size=1)

        self.center = nn.Sequential(DualAttention(512 * 4, 256),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, batch):
        
        conv1 = self.conv1(batch['data'])
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center_512 = self.center_global_pool(conv5)
        center_64 = self.center_conv1x1(center_512)

        f = self.center(conv5)

        out = {}
        
        out['conv2'] = conv2
        out['conv3'] = conv3
        out['conv4'] = conv4
        out['conv5'] = conv5
        out['f'] = f
        out['center_64'] = center_64
        
        return out
