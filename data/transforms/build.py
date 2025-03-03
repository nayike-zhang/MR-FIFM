# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T
from .transforms import RandomErasing
import numpy as np



def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    if is_train:
        transform = T.Compose([  # 用于将多个变换组合在一起
            # T.Resize(cfg.INPUT.SIZE_TRAIN),  # 调整图像大小为训练阶段指定的尺寸[128, 128]
            # T.RandomHorizontalFlip(p=0.5),  # 以指定的概率进行随机水平翻转0.5
            # T.RandomVerticalFlip(p=0.5),
            # T.Pad(cfg.INPUT.PADDING),  # 使用填充大小10进行填充
            # T.RandomCrop(cfg.INPUT.SIZE_TRAIN),  # 随机裁剪到训练阶段指定的尺寸
            T.ToTensor(),  # 转换为张量形式
            normalize_transform,  # 应用之前创建的归一化变换
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN) # 使用随机擦除进行数据增强0.4，擦除区域的像素值使用均值填充[0.485, 0.456, 0.406]
        ])
    else:
        transform = T.Compose([
            # T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])

    return transform
