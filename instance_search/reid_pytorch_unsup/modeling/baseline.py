# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.detnet import detnet,detnet_large,detnet_small
from .backbones.resnet_ibn_a import resnet50_ibn_a,resnet101_ibn_a,resnet50_ibn_a_LeakyRelu
import torch.nn.functional as F

class GeM(nn.Module):

    def __init__(self, p=3.0, eps=1e-6, freeze_p=True):
        super(GeM, self).__init__()
        self.p = p if freeze_p else Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                            (1, 1)).pow(1. / self.p)

    def __repr__(self):
        if isinstance(self.p, float):
            p = self.p
        else:
            p = self.p.data.tolist()[0]
        return self.__class__.__name__ +\
               '(' + 'p=' + '{:.4f}'.format(p) +\
               ', ' + 'eps=' + str(self.eps) + ')'

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    # NOTE: hacky here
    # torch.manual_seed(0)
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck_planes, dropout_rate, model_name, pretrain_choice, loss_type, task_prob, gem_pooling):
        super(Baseline, self).__init__()
        self.model_name = model_name
        self.loss_type = loss_type
        self.task_prob = task_prob
        self.num_classes = num_classes
        self.gem_pooling = gem_pooling

        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
            
        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 6, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 23, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)  
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=64, 
                              reduction=16,
                              dropout_p=0.2, 
                              last_stride=last_stride)
        elif model_name == 'resnet50_ibn_a':
            self.in_planes = 2048
            self.base = resnet50_ibn_a(last_stride)
            print('using resnet50_ibn_a as a backbone')
        elif model_name == 'detnet':
            self.in_planes = 1024
            self.base = detnet()
        elif model_name == 'detnet_large':
            self.in_planes = 1024
            self.base = detnet_large()
        elif model_name == 'detnet_small':
            self.in_planes = 512
            self.base = detnet_small()
        if self.gem_pooling:
            print('using GeM pooling')
            self.gem = GeM()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.neck_planes = neck_planes
        self.dropout_rate = dropout_rate

        if self.neck_planes > 0:
            self.fcneck = nn.Linear(self.in_planes, self.neck_planes, bias=False)
            self.fcneck.apply(weights_init_xavier)
            self.fcneck_bn = nn.BatchNorm1d(self.neck_planes)
            self.fcneck_bn.apply(weights_init_kaiming)
            self.in_planes = self.neck_planes
            #print('fcneck is used.')

            self.relu = nn.ReLU(inplace=True)

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
            #print('dropout is used: %f.' %self.dropout_rate)

        assert(len(self.loss_type)==len(self.num_classes))

        # source data
        i = 0
        if 'cls' in self.loss_type[i] and self.num_classes[i] != -1 and self.task_prob[i] > 0.:
            self.src_classifier = nn.Linear(self.in_planes, self.num_classes[i], bias=False)
            self.src_classifier.apply(weights_init_classifier)

        # target sup data
        i = 1
        if 'cls' in self.loss_type[i] and self.num_classes[i] != -1 and self.task_prob[i] > 0.:
            self.tgt_sup_classifier = nn.Linear(self.in_planes, self.num_classes[i], bias=False)
            self.tgt_sup_classifier.apply(weights_init_classifier)

        # target unsup data
        i = 2
        if 'cls' in self.loss_type[i] and self.num_classes[i] != -1 and self.task_prob[i] > 0.:
            self.tgt_unsup_classifier = nn.Linear(self.in_planes, self.num_classes[i], bias=False)
            self.tgt_unsup_classifier.apply(weights_init_classifier)

    def forward(self, x):

        x = self.base(x)
        if self.gem_pooling:
            global_feat = self.gem(x)
        else:
            global_feat = self.gap(x) + self.gmp(x)  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        # global_feat = global_feat.view(1, 2048)  # flatten to (bs, 2048)

        if self.neck_planes > 0:
            global_feat = self.fcneck(global_feat)
            global_feat = self.fcneck_bn(global_feat)
            #global_feat = self.relu(global_feat)
        if self.dropout_rate > 0:
            global_feat = self.dropout(global_feat)

        if self.training:
            outs = []
            for i in range(len(self.loss_type)):
                if 'cls' in self.loss_type[i] and self.num_classes[i] != -1 and self.task_prob[i] > 0.:
                    if i == 0:
                        out = self.src_classifier(global_feat)
                    elif i == 1:
                        out = self.tgt_sup_classifier(global_feat)
                    elif i == 2:
                        out = self.tgt_unsup_classifier(global_feat)
                    outs.append(out)
                else:
                    outs.append(torch.tensor(0.,requires_grad=True).cuda())
            outs.append(global_feat)
            return outs
        else:
            return global_feat

    def load_param(self, trained_path, pretrain_choice):
        if pretrain_choice == 'self':
            param_dict = torch.load(trained_path, map_location='cpu')
            for i in param_dict:
                if 'classifier' in i:
                    #if 'src_classifier.weight' in self.state_dict():
                    #    self.state_dict()['src_classifier.weight'].copy_(param_dict[i])
                    continue
                self.state_dict()[i].copy_(param_dict[i])
        elif pretrain_choice == 'imagenet':
            self.base.load_param(trained_path)
