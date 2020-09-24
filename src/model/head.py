import torch
import torch.nn as nn
import torch.nn.functional as F


class SCSEBlock(nn.Module):
    
    '''
    Squeeze Excitation block
    paper: https://arxiv.org/abs/1709.01507

    Input & Output
        Input: [Float Tensor]
        Output: [Float Tensor]
    '''
        
    def __init__(self, channel, reduction=16):
        
        super(SCSEBlock, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel // reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel // reduction), channel),
                                                nn.Sigmoid())

        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())

    def forward(self, x):
        
        bahs, chs, _, _ = x.size()

        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = self.channel_excitation(chn_se).view(bahs, chs, 1, 1)
        chn_se = torch.mul(x, chn_se)

        spa_se = self.spatial_se(x)
        spa_se = torch.mul(x, spa_se)
        
        return torch.add(chn_se, 1, spa_se)


class Decoder(nn.Module):
    
    '''
    Decoder for Unet-like structure

    Args:
        in_channels: [int]
        channels: [int]
        out_channels: [int]
    '''

    def __init__(self, in_channels, channels, out_channels):
        
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(channels),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(channels, out_channels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True))
        self.SCSE = SCSEBlock(out_channels)

    def forward(self, x, e=None):
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if e is not None:
            x = torch.cat([x, e], 1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.SCSE(x)
        
        return x


class DA_SCSE_ResNeXt50_Decoder(nn.Module):
    
    '''
    Unet-like deocder for DA_SCSE_ResNeXt50 (Bottleneck) decoder + Hypercolumn

    Args:
        num_classes: [int], number of output classes
        fc_classes: [int], number of auxiliary classification labels

    Input & Output:
        Input:
            - features: [dict], with keys "conv2", "conv3", "conv4", "conv5" ....
        Output:
            - out: [dict], with key - "x_final"...
    '''


    def __init__(self, num_classes=3, fc_classes=2):
        
        super(DA_SCSE_ResNeXt50_Decoder, self).__init__()

        self.num_classes = num_classes
        self.fc_classes = fc_classes

        self.decoder5 = Decoder(256 + 512 * 4, 512, 64)
        self.decoder4 = Decoder(64 + 256 * 4, 256, 64)
        self.decoder3 = Decoder(64 + 128 * 4, 128, 64)
        self.decoder2 = Decoder(64 + 64 * 4, 64, 64)
        self.decoder1 = Decoder(64, 32, 64)

        self.logits_no_empty = nn.Sequential(nn.Conv2d(320, 64, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0))

        self.logits_final = nn.Sequential(nn.Conv2d(320 + 64, 64, kernel_size=3, padding=1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0))
        if self.fc_classes > 0:
            self.center_fc = nn.Linear(64, fc_classes)

    def forward(self, features):
        
        conv2, conv3, conv4, conv5 = features['conv2'], features['conv3'], features['conv4'], features['conv5']
        f, center_64 = features['f'], features['center_64']

        d5 = self.decoder5(f, conv5)
        d4 = self.decoder4(d5, conv4)
        d3 = self.decoder3(d4, conv3)
        d2 = self.decoder2(d3, conv2)
        d1 = self.decoder1(d2)

        hypercol = torch.cat((
            d1,
            F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False),
            F.interpolate(d3, scale_factor=4, mode='bilinear', align_corners=False),
            F.interpolate(d4, scale_factor=8, mode='bilinear', align_corners=False),
            F.interpolate(d5, scale_factor=16, mode='bilinear', align_corners=False)), 1)

        x_no_empty = self.logits_no_empty(hypercol)
        
        hypercol_add_center = torch.cat((
            hypercol,
            F.interpolate(center_64, scale_factor=256, mode='bilinear', align_corners=False)), 1)

        x_final = self.logits_final(hypercol_add_center)

        out = {}
        if self.fc_classes > 0:
            center_64_flatten = center_64.view(center_64.size(0), -1)
            center_fc = self.center_fc(center_64_flatten)
            out['center_fc'] = center_fc
            
        out['x_no_empty'] = x_no_empty
        out['x_final'] = x_final

        return out
