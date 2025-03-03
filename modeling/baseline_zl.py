# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from functools import reduce
import operator
import pdb
import torch
from torch import nn
from torchvision import models

from modeling.resnet_zl import ResNet18, ResNet_1, ResNet50_share, ResNet_share, ResNet_p, ECABasicBlock
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
# from .backbones.squeezenet import SqueezeNet,Fire
# from .backbones.densenet import _DenseLayer, _DenseBlock, _Transition, DenseNet
from .backbones.mobilenet import ConvBNReLU, InvertedResidual, MobileNetV2
from .backbones.inception import Inception3, BasicConv2d
from .transformer import myViT, Rearrange
from .attention import eca_layer, AFF, GatedFusion, SE_Block, GatedMultimodalLayer, ModalityFusionGate, eca_layer1


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
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def conv3x3(in_plane, out_plane, stride=1, bn=True, relu=True):
    layer = []
    if bn:
        bias_flag = False
    else:
        bias_flag = True
    layer.append(nn.Conv2d(in_channels=in_plane, out_channels=out_plane, kernel_size=(3, 3),
                           stride=stride, padding=1, bias=bias_flag))
    if bn:
        layer.append(nn.BatchNorm2d(out_plane))
    if relu:
        layer.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layer)


def conv1x1(in_plane, out_plane, bn=False, activation=False):
    layer = []
    if bn:
        bias_flag = False
    else:
        bias_flag = True
    layer.append(nn.Conv2d(in_channels=in_plane, out_channels=out_plane, kernel_size=(1, 1), bias=bias_flag))
    if bn:
        layer.append(nn.BatchNorm2d(out_plane))
    if activation == 'relu':
        layer.append(nn.ReLU())
    elif activation == 'sigmoid':
        layer.append(nn.Sigmoid())
    elif activation == 'leakyrelu':
        layer.append(nn.LeakyReLU())
    return nn.Sequential(*layer)


class ChannelAttModule(nn.Module):
    def __init__(self, in_planes, out_plane, reduction=4):
        super(ChannelAttModule, self).__init__()
        self.downsample = nn.Sequential(
            conv3x3(in_plane=in_planes, out_plane=in_planes),
            conv1x1(in_plane=in_planes, out_plane=out_plane),
            conv3x3(in_plane=out_plane, out_plane=out_plane)
        )
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv1x1(in_plane=out_plane, out_plane=out_plane // reduction, activation='relu'),
            conv1x1(in_plane=out_plane // reduction, out_plane=out_plane),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.downsample(x)
        att = self.attention(feat)
        return feat * att + feat


class ASU(nn.Module):  # ALNU中计算缩放、平移因子的部分
    def __init__(self, in_dim=512):
        super(ASU, self).__init__()
        self.conv1 = nn.Sequential(
            conv3x3(in_plane=in_dim, out_plane=in_dim // 4, stride=1, bn=True, relu=True),
            conv3x3(in_plane=in_dim // 4, out_plane=in_dim // 16, stride=1, bn=True, relu=True)
        )
        self.GAP = nn.AdaptiveAvgPool2d(output_size=1)
        self.GMP = nn.AdaptiveMaxPool2d(output_size=1)
        self.conv_alpha = conv1x1(in_plane=in_dim // 8, out_plane=1, bn=False, activation='sigmoid')
        # self.conv_beta = conv1x1(in_plane=in_dim//8, out_plane=1, bn=False, activation='sigmoid')

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([self.GAP(x), self.GMP(x)], dim=1)
        alpha = self.conv_alpha(x)
        # beta = self.conv_beta(x)
        # print(alpha)
        return alpha  # , beta


class ModalNorm(nn.Module):  # ALNU
    def __init__(self, dim=512):
        super(ModalNorm, self).__init__()

        # self.gama = nn.Parameter(torch.ones(1))
        # self. beta = nn.Parameter(torch.zeros(1))
        # self.sigmoid = nn.Sigmoid()

        self.conv_gamma = ASU(dim)
        self.conv_beta = ASU(dim)

    def forward(self, x):
        # target_mean = self.sigmoid(self.sigma)
        # var = ((X - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        cur_mean = x.mean(dim=[1, 2, 3], keepdim=True)
        cur_var = ((x - cur_mean) ** 2).mean(dim=[1, 2, 3], keepdim=True)
        x_hat = (x - cur_mean) / torch.sqrt(cur_var + 1e-6)

        gamma = self.conv_gamma(x)
        beta = self.conv_beta(x)

        f = gamma * x_hat + beta
        # pdb.set_trace()
        return f, x


class Baseline(nn.Module):
    in_planes = 512

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Baseline, self).__init__()
        self.model_name = model_name
        print(self.model_name)
        if model_name == 'resnet18':
            self.in_planes = 128
            self.base = ResNet18(last_stride=last_stride, block=BasicBlock, layers=[2, 2])
        elif model_name == 'resnet50':
            print("------------------------------------------------------------------------------")
            self.in_planes = 512
            self.base = ResNet18(last_stride=last_stride, block=Bottleneck, layers=[3, 4])
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
            self.modalnorm = ModalNorm()
        elif model_name == 'mobilenet':
            self.in_planes = 1280
            self.base = MobileNetV2()
            self.modalnorm = ModalNorm(160)
        else:
            raise Exception('unsupported model')

        if pretrain_choice == 'imagenet':
            print("1111111111111")
            # self.base.load_param(model_path)
            # print('1not Loading pretrained ImageNet model......')
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        # self.modalnorm = ModalNorm()

        if self.neck == 'no':
            # pass
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift

            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        if self.model_name == "resnet50":
            x = self.base.conv1(x)
            x = self.base.bn1(x)
            x = self.base.maxpool(x)
            x = self.base.layer1(x)
            x = self.base.layer2(x)
            # x, _ = self.modalnorm(x)
            # x = self.base.layer3(x)
            # x = self.base.layer4(x)
        elif self.model_name == "resnet18":
            x = self.base.conv1(x)
            x = self.base.bn1(x)
            # x = self.base.relu(x)
            x = self.base.maxpool(x)
            x = self.base.layer1(x)
            x = self.base.layer2(x)
            # x = self.base.layer3(x)
            # x = self.base.layer4(x)

        elif self.model_name == "se_resnet50":
            x = self.base.layer0(x)
            x = self.base.layer1(x)
            x = self.base.layer2(x)
            x._ = self.modalnorm(x)
            x = self.base.layer3(x)
            x = self.base.layer4(x)
        elif self.model_name == 'mobilenet':
            layers = self.base.features
            for i in range(len(layers)):
                if i == len(layers) - 3:
                    x, _ = self.modalnorm(x)
                x = layers[i](x)
        else:
            raise NotImplementedError

        p_feat = self.gap(x)  # B, Channel, 1, 1
        p_feat = p_feat.view(p_feat.shape[0], -1)  # shape = [32, 2048]

        if self.neck == 'bnneck':

            pfeat = self.bottleneck(p_feat)  # shape = [32, 2048]
        else:
            pfeat = p_feat

        p_score = self.classifier(pfeat)  # shape = [32, 300]

        return x, p_feat, pfeat, p_score, 0, x.mean(dim=[2, 3])
        # return global_feat, feat, score, 0, x.mean(dim=[2, 3])

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        net_params_keys = self.state_dict().keys()
        for key in net_params_keys:
            if 'num_batches_tracked' in key:
                continue
            if key[5:] not in param_dict:
                continue
            self.state_dict()[key].copy_(param_dict[key[5:]])


class MainNet(nn.Module):
    in_planes = 512

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, branches=3):
        super(MainNet, self).__init__()
        self.in_planes = 512
        self.branches = branches
        self.num_classes = num_classes

        self.base1 = ResNet_share(last_stride=last_stride, block=BasicBlock, layers=[2, 2])
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.conv1 = conv1x1(in_plane=128, out_plane=128, bn=True, activation=False)
        self.conv2 = conv1x1(in_plane=512, out_plane=512, bn=True, activation=False)
        self.conv3 = conv1x1(in_plane=256, out_plane=256, bn=True, activation=False)
        self.conv4 = conv1x1(in_plane=1024, out_plane=1024, bn=True, activation=False)
        self.conv5 = conv1x1(in_plane=1536, out_plane=1536, bn=True, activation=False)
        self.fc = nn.Sequential(
            #nn.LayerNorm(1024),
            nn.Linear(1024, num_classes)
        )
        # self.attention = MultiheadAttention(hid_dim=512, n_heads=2, dropout=0.1)

        self.myvit = myViT(image_size=16, patch_size=2, num_classes=num_classes, dim=512, depth=3, heads=8, mlp_dim=512*4, dropout=0.1, emb_dropout=0.02)

        self.eca = eca_layer1(channel=1024, k_size=3)
        # self.fusion_module = GatedMultimodalLayer(512, 512, 300)
        # self.fusion_module = ModalityFusionGate(512, 512)
        # self.aff_module = AFF(channels=512, r=2)
        self.restore = Rearrange('b (h w) c -> b c h w', w=1)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)


        for i in range(branches):
            self.__setattr__('branch_' + str(i),
                             Baseline(num_classes, last_stride, model_path, neck, neck_feat, model_name,
                                      pretrain_choice))

        self.fuse_classifier = nn.Linear(1280, num_classes)
        self.fuse_classifier.requires_grad_(True)
        print("*" * 20, self.fuse_classifier)

    def forward(self, inputs):

        weight_list = []  # 没有用，懒得删了，可以用来存放一些中间结果啥的

        s_list = []  # saving global features for final representation and loss computing
        p_list = []
        bn_p_list = []
        bn_s_list = []
        pred_p_list = []
        pred_s_list = []
        mid_list = []
        x_list = []
        x1_list = []
        for i in range(self.branches):
            x, p_feat, pfeat, p_score, w, mid_feat = self.__getattr__('branch_' + str(i)).forward(inputs[i])  # stage 2 for branches
            x1 = self.conv1(x)
            x_list.append(x1)

            p_list.append(p_feat)
            bn_p_list.append(pfeat)
            pred_p_list.append(p_score)
            mid_list.append(mid_feat)

            x = self.base1.layer3(x)
            x = self.base1.layer4(x)
            x2 = self.conv2(x)
            x1_list.append(x2)

            s_feat = self.gap(x)  # B, Channel, 1, 1
            s_feat = s_feat.view(s_feat.shape[0], -1)  # shape = [32, 2048]

            sfeat = self.bottleneck(s_feat)  # shape = [32, 2048]
            s_score = self.classifier(sfeat)

            s_list.append(s_feat)
            bn_s_list.append(sfeat)
            pred_s_list.append(s_score)

        x1_list0 = nn.functional.adaptive_avg_pool2d(input=x1_list[0], output_size=(1, 1))
        x1_list1 = nn.functional.adaptive_avg_pool2d(input=x1_list[1], output_size=(1, 1))
        x1, x2 = self.myvit(x1_list0, x1_list1)

        x1 = self.restore(x1)
        x2 = self.restore(x2)
        # # x = self.restore(x3)

        x = torch.cat((x1, x2), dim=1)
        #x = torch.cat((x1_list0, x1_list1), dim=1)

        x = self.conv4(x)
        x = self.eca(x)

        #x = nn.functional.adaptive_avg_pool2d(input=x, output_size=(1, 1))
        fuse_feat = torch.flatten(x, start_dim=1)
        fuse_score = self.fc(fuse_feat)

        if self.training:
            return pred_s_list, pred_p_list, s_list, p_list, weight_list, bn_s_list, bn_p_list, [fuse_score], fuse_feat
        else:
            return pred_s_list, pred_p_list, s_list, p_list, weight_list, bn_s_list, bn_p_list, [fuse_score], fuse_feat  # return gf_list, weight_list, bn_f_list, mid_list, [fuse_score], fuse_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        # pdb.set_trace()
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
        # pdb.set_trace()
        out += residual
        out = self.relu(out)

        return out