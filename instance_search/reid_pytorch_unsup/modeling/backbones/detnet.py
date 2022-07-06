import torch
import torch.nn as nn


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, groups=1, dilation=1):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                    #nn.AvgPool2d(kernel_size=3, stride=stride, count_include_pad=False, padding=1),
                    conv1x1(inplanes, planes, stride),
                    nn.BatchNorm2d(planes),
            )

        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class BasicBlock2(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, groups=1, dilation=1):
        super(BasicBlock2, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=3, stride=stride, count_include_pad=False, padding=1),
                    conv1x1(inplanes, planes, 1),
                    nn.BatchNorm2d(planes),
            )

        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, groups=1, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes//2)
        self.bn1 = nn.BatchNorm2d(planes//2)
        self.conv2 = conv3x3(planes//2, planes//2, stride, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes//2)
        self.conv3 = conv1x1(planes//2, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                    #nn.AvgPool2d(kernel_size=3, stride=stride, count_include_pad=False, padding=1),
                    conv1x1(inplanes, planes, stride),
                    nn.BatchNorm2d(planes),
            )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck2(nn.Module):
    def __init__(self, inplanes, planes, stride=1, groups=1, dilation=1):
        super(Bottleneck2, self).__init__()
        self.conv1 = conv1x1(inplanes, planes//4)
        self.bn1 = nn.BatchNorm2d(planes//4)
        self.conv2 = conv3x3(planes//4, planes//4, stride, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes//4)
        self.conv3 = conv1x1(planes//4, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                    #nn.AvgPool2d(kernel_size=3, stride=stride, count_include_pad=False, padding=1),
                    conv1x1(inplanes, planes, stride),
                    nn.BatchNorm2d(planes),
            )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class DetNet(nn.Module):
    def __init__(self, in_channels_group, channels_group, strides, block_type, classes=1000, name='normal', zero_init_residual=False):
        super(DetNet, self).__init__()
        block = {'res' : BasicBlock,
                 'res2': BasicBlock2,
                 'bottleneck' : Bottleneck,
                 'bottleneck2' : Bottleneck2}
        self.conv1 = conv3x3(3, 32, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.name = name
        if self.name == 'small':
            self.small_layer = nn.Sequential(conv3x3(32, 64, stride=2),
                                             nn.BatchNorm2d(64),
                                             nn.ReLU(inplace=True))
        elif self.name == 'large':
            self.extra_layer = nn.Sequential(conv3x3(32, 128, stride=2),
                                             nn.BatchNorm2d(128),
                                             nn.ReLU(inplace=True))
                                             
        layers = []
        for in_c, c, s, b in zip(in_channels_group, channels_group, strides, block_type):
            assert b in ['res', 'res2', 'bottleneck', 'bottleneck2']
            layers.append(block[b](inplanes=in_c, planes=c, stride=s))
        self.body = nn.Sequential(*layers)

        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(channels_group[-1], classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) or isinstance(m, Bottleneck2):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.name == 'small':
            x = self.small_layer(x)

        x = self.body(x)
        '''
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        '''
        return x

    def load_param(self, model_path):
        '''
        param_dict = torch.load(model_path)['state_dict']
        for i in param_dict:
            print('train:',i)
        for i in self.state_dict():
            print('model:',i)
        vvv
        '''
        param_dict = torch.load(model_path, map_location='cpu')['state_dict']
        for i in param_dict:
            if 'fc' in i:
                continue
            if 'body.15.downsample' in i and 'small' in self.name:
                continue
            if 'body.21.downsample' in i and 'normal' in self.name:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    forward = _forward


def _detnet(in_channels_group, channels_group, strides, block_type, classes, name='normal'):
    model = DetNet(in_channels_group, channels_group, strides, block_type, classes=classes, name=name)
    return model


def detnet(pretrained=False, classes=1000):
    in_channels_group = [32] + [32] + [128] * 3 + [128] * 3 + [256] * 3 + [512] * 10 + [1024] * 6
    channels_group = [32] + [128] * 3 + [128] * 3 + [256] * 3 + [512] * 10 + [1024] * 7
    strides = [1] + [2] + [1] * 3 + [2] + [1] * 4 + [2] + [1] * 10 + [1] + [1] * 5
    block_type = ['res'] * 10 + ['bottleneck2'] * 15 + ['bottleneck'] * 2
    return _detnet(in_channels_group, channels_group, strides, block_type, classes)


def detnet_small(pretrained=False, classes=1000):
    channels_group = [64] * 4 + [128] * 10 + [256] * 5 + [512] * 2
    strides = [2] + [1] * 3 + [2] + [1] * 10 + [1] + [1] * 5
    block_type = ['res2'] * 20
    return _detnet(channels_group[:-1], channels_group[1:], strides, block_type, classes, name='small')
    
def detnet_large(pretrained=False, classes=1000):
    channels_group = [128] * 8 + [256] * 9 + [512] * 14 + [1024] * 6
    strides = [1] * 7 + [2] + [1] * 8 + [2] + [1] * 13 + [2] + [1] * 5
    block_type = ['res'] * 16 + ['bottleneck2'] * 18 + ['bottleneck'] * 2
    return _detnet(channels_group[:-1], channels_group[1:], strides, block_type, classes, name='large')


