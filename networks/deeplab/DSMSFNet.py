import torch
import torch.nn as nn
import torch.nn.functional as F
from ..common_func.get_backbone import get_model
from ..common_func.get_backbone import get_nsdm_model
from .deeplabv3 import _ASPP
from ..common_func.base_func import _ConvBNReLU

class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)

class _ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rate, norm_layer):
        super(_ASPPConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

class _AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(_AsppPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out

class _AsppMaxPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(_AsppMaxPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out

class _MSF(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer):
        super(_MSF, self).__init__()
        out_channels = 256
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer)

        self.fus0 = nn.Sequential(
            nn.Conv2d(4 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

        self.fus1 = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

        self.b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer)
        self.b5 = _AsppMaxPooling(in_channels, out_channels, norm_layer=norm_layer)


        self.project = nn.Sequential(
            nn.Conv2d(out_channels*4, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

        self.conv=nn.Conv2d(512, 256, 3, padding=1)
        self.relu=nn.ReLU()

    def forward(self, x):
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)

        fusion0 = self.fus0(torch.cat((feat1, feat2, feat3, feat4), dim=1))
        fusion1 = self.fus1(torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1))

        # feat6 = self.b5(x)
        # fusion->256 feat->256 x 48
        # x = torch.cat(( fusion, feat5), dim=1)
        x = torch.cat((feat1, fusion0, fusion1, feat5), dim=1)
        x = self.project(x)
        return x

class SELayer(nn.Module):

    def __init__(self, inplanes, planes, reduction=16):
        super(SELayer, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(inplanes, planes // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(planes // reduction, planes)
        self.sigmoid = nn.Sigmoid()
        self.planes = planes

    def forward(self, x):
        batch, channel, _, _ = x.size()
        x = self.global_pool(x)
        x = x.view(batch, channel)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x.view(batch, self.planes, 1, 1)
        return x

class SelectiveFusionModule(nn.Module):
    def __init__(self, inplanes, planes, reduction=16):
        super(SelectiveFusionModule, self).__init__()
        self.se = SELayer(inplanes, planes, reduction)

    # x1->irrg feature x2->dsm feature

    def forward(self, x1, x2):
        out = torch.cat([x1, x2], dim=1)
        se = self.se(out)
        out = x1 * se + x1
        return out

class DSMSFNet(nn.Module):

    def __init__(self, nclass, **kwargs):
        super(DSMSFNet, self).__init__()
        aux = True
        dilated = True
        self.aux = aux
        self.nclass = nclass
        self.drop_path_prob = 0.0
        """
            dual source feature fusion structure
        """
        # EfficientNet backbone
        self.pretrained = get_model('efficientnet_b2')
        self.nsdm_pretrained = get_nsdm_model('efficientnet_b2')

        self.leakyrelu= nn.LeakyReLU()
        self.head = _DSMSFNetHead(nclass, c1_channels=24, **kwargs)


        # Selective Fusion Module
        self.sfm1 = SelectiveFusionModule(32, 16, reduction=8)
        self.bn1 = nn.BatchNorm2d(16)
        self.sfm2 = SelectiveFusionModule(48, 24, reduction=8)
        self.bn2 = nn.BatchNorm2d(24)
        self.sfm3 = SelectiveFusionModule(96, 48, reduction=8)
        self.bn3 = nn.BatchNorm2d(48)
        self.sfm4 = SelectiveFusionModule(240, 120, reduction=8)
        self.bn4 = nn.BatchNorm2d(120)
        self.sfm5 = SelectiveFusionModule(704, 352, reduction=8)
        self.bn5 = nn.BatchNorm2d(352)

        self.deconv1 = nn.Conv2d(6, 6, 3, padding=1)
        self.debn1 = nn.BatchNorm2d(6)
        self.relu = nn.ReLU()

        self.block = nn.Sequential(
            _ConvBNReLU(256, 256, 3, padding=1, norm_layer=nn.BatchNorm2d),
            nn.Dropout(0.1),
            nn.Conv2d(256, nclass, 1))

        if aux:
            self.auxlayer = _FCNHead(728, nclass)


    # x-> irrg image and y-> ndsm data
    def base_forward(self ,x ,y):
        """
            dual source feature fusion structure
        """

        #  EfficientNet backbone
        features = self.pretrained(x)
        nsdm_feature=self.nsdm_pretrained(y)

        # Selective Fusion Module
        features[0] = self.bn1(self.relu(self.sfm1(features[0], nsdm_feature[0])))
        features[1] = self.bn2(self.relu(self.sfm2(features[1], nsdm_feature[1])))
        features[2] = self.bn3(self.relu(self.sfm3(features[2], nsdm_feature[2])))
        features[3] = self.bn4(self.relu(self.sfm4(features[3], nsdm_feature[3])))

        return features[0], features[1], features[2], features[3]

    def forward(self, x, y):

        size_x = x.size()[2:]

        # output of the dual source feature fusion structure
        c0, c1, c2, c3  = self.base_forward(x,y)

        # multi-scale feature fusion modules and decoder fusion structure
        x = self.head(c3, c2, c1, c0)
        x = F.interpolate(x, size_x, mode='bilinear', align_corners=True)

        x = self.block(x)

        return x


class _DSMSFNetHead(nn.Module):
    def __init__(self, nclass, c1_channels=128, norm_layer=nn.BatchNorm2d):
        super(_DSMSFNetHead, self).__init__()
        self.msf1 = _MSF(48, [3,6,9], norm_layer)
        self.msf2 = _MSF(120, [3,6,9], norm_layer)

        self.c1_block = _ConvBNReLU(c1_channels, 48, 3, padding=1, norm_layer=norm_layer)
        self.c2_block = _ConvBNReLU(16, 48, 3, padding=1, norm_layer=norm_layer)

        self.block1 = nn.Sequential(
            _ConvBNReLU(304, 256, 3, padding=1, norm_layer=norm_layer),
            nn.Dropout(0.5),
            _ConvBNReLU(256, 256, 3, padding=1, norm_layer=norm_layer))

        self.block2 = nn.Sequential(
            _ConvBNReLU(304, 256, 3, padding=1, norm_layer=norm_layer),
            nn.Dropout(0.5),
            _ConvBNReLU(256, 256, 3, padding=1, norm_layer=norm_layer))

        self.block3 = nn.Sequential(
            _ConvBNReLU(512, 256, 3, padding=1, norm_layer=norm_layer),
            nn.Dropout(0.5),
            _ConvBNReLU(256, 256, 3, padding=1, norm_layer=norm_layer)
            # nn.Dropout(0.1)
        )

        #self.deconv1 = nn.Conv2d(256, 256, 3, padding=1)
        self.debn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

    def forward(self, c3, c2, c1, c0):

        size = c1.size()[2:]
        c1 = self.c1_block(c1)
        c0 = self.c2_block(c0)

        # multi-scale feature fusion modules
        c2 = self.msf1(c2)
        c3 = self.msf2(c3)

        # decoder fusion structure
        x1 = F.interpolate(c3, [128, 128], mode='bilinear', align_corners=True)
        x1 = torch.cat([x1, c1], dim=1)
        x1 = self.block1(x1)
        x1 = F.interpolate(x1, [256, 256], mode='bilinear', align_corners=True)
        x2 = F.interpolate(c2, [256, 256], mode='bilinear', align_corners=True)
        x2 = torch.cat([x2, c0], dim=1)
        x2 = self.block2(x2)

        x3 = torch.cat([x2, x1], dim=1)
        x3 = self.block3(x3)

        return x3

if __name__ == '__main__':
    net = DSMSFNet(nclass=6)
    x = torch.randn(2, 3, 256, 256)
    y = net(x)
    print(y.shape)
