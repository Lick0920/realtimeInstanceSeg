# Copyright (c) Tianheng Cheng and its affiliates. All Rights Reserved

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

from detectron2.utils.registry import Registry
from detectron2.layers import Conv2d

SPARSE_INST_ENCODER_REGISTRY = Registry("SPARSE_INST_ENCODER")
SPARSE_INST_ENCODER_REGISTRY.__doc__ = "registry for SparseInst decoder"


class PyramidPoolingModule(nn.Module):

    def __init__(self, in_channels, channels=512, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, channels, size) for size in sizes]
        )
        self.bottleneck = Conv2d(
            in_channels + len(sizes) * channels, in_channels, 1)

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = Conv2d(features, out_features, 1)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=F.relu_(stage(feats)), size=(
            h, w), mode='bilinear', align_corners=False) for stage in self.stages] + [feats]
        out = F.relu_(self.bottleneck(torch.cat(priors, 1)))
        return out



@SPARSE_INST_ENCODER_REGISTRY.register()
class InstanceContextEncoder(nn.Module):
    """ 
    Instance Context Encoder
    1. construct feature pyramids from ResNet
    2. enlarge receptive fields (ppm)
    3. multi-scale fusion 
    """

    def __init__(self, cfg, input_shape):
        super().__init__()
        self.num_channels = cfg.MODEL.SPARSE_INST.ENCODER.NUM_CHANNELS
        self.in_features = cfg.MODEL.SPARSE_INST.ENCODER.IN_FEATURES
        self.in_channels = [input_shape[f].channels for f in self.in_features]
        fpn_laterals = []
        fpn_outputs = []
        for in_channel in reversed(self.in_channels):
            lateral_conv = Conv2d(in_channel, self.num_channels, 1)
            output_conv = Conv2d(self.num_channels, self.num_channels, 3, padding=1)
            c2_xavier_fill(lateral_conv)
            c2_xavier_fill(output_conv)
            fpn_laterals.append(lateral_conv)
            fpn_outputs.append(output_conv)
        self.fpn_laterals = nn.ModuleList(fpn_laterals)
        self.fpn_outputs = nn.ModuleList(fpn_outputs)
        # ppm
        self.ppm = PyramidPoolingModule(self.num_channels, self.num_channels // 4)
        # final fusion # 0411 lck 
        # self.fusion = nn.Conv2d(self.num_channels * 3, self.num_channels, 1)
        # # 0414 lck 3d fusion 有没有道理？
        # self.fusion_3d1 = nn.Conv3d(self.num_channels, self.num_channels, 1,stride=(2,1,1))
        # self.fusion_3d2 = nn.Conv3d(self.num_channels, self.num_channels, 1,stride=(2,1,1))
        # c2_msra_fill(self.fusion_3d1)
        # c2_msra_fill(self.fusion_3d2)
        self.fusion = nn.Conv2d(self.num_channels * 4, self.num_channels, 1) # 新增spd分支
        c2_msra_fill(self.fusion)

    def forward(self, features):
        features = [features[f] for f in self.in_features]
        # for i in range(len(features)):
        #     print("features shape",features[i].shape)  

        features = features[::-1]

        prev_features = self.ppm(self.fpn_laterals[0](features[0]))
        # print("prev_features shape",prev_features.shape)
        outputs = [self.fpn_outputs[0](prev_features)]
        # print("outputs shape",outputs[0].shape)
        i = 0
        for feature, lat_conv, output_conv in zip(features[1:], self.fpn_laterals[1:], self.fpn_outputs[1:]):
            i = i+1
            # print("before  lat_conv feature shape",feature.shape)
            lat_features = lat_conv(feature)
            # print("after  lat_conv feature shape",lat_features.shape)
            # print("prev_features shape", prev_features.shape)
            ####### up 4
            # if feature.shape[2] > 96:
            #     top_down_features = F.interpolate(prev_features, scale_factor=4.0, mode='nearest')
            # else:
            #     top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode='nearest')
            #### lck 0414 5层的时候第一层不用up ###############
            if i == 4:
                top_down_features = F.interpolate(prev_features, scale_factor=1.0, mode='nearest')
            else:
                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode='nearest')
            # print("top_down_features shape",top_down_features.shape)
            
            prev_features = lat_features + top_down_features
            outputs.insert(0, output_conv(prev_features))
        size = outputs[0].shape[2:]
        features = [
            outputs[0]] + [F.interpolate(x, size, mode='bilinear', align_corners=False) for x in outputs[1:]] # [3, 256, 384, 256]
        
        # 0414 lck 3dfusion 有没有道理？
        # features = self.fusion_3d1(torch.stack(features,dim=2))
        # print('after fusion_3d1',features.shape)
        # features = self.fusion_3d2(features).squeeze(2)
        features = self.fusion(torch.cat(features, dim=1))  # [1, 256, 384, 256]) 直接下采样3倍
        print('after encoder',features.shape)
        return features


def build_sparse_inst_encoder(cfg, input_shape):
    name = cfg.MODEL.SPARSE_INST.ENCODER.NAME
    return SPARSE_INST_ENCODER_REGISTRY.get(name)(cfg, input_shape)


# if __name__ == "__main__":
#     import torch
#     from detectron2.config import get_cfg
#     from detectron2.modeling import build_model
#     from detectron2.checkpoint import DetectionCheckpointer
#     from detectron2.engine import default_argument_parser, default_setup
    
#     SPARSE_INST_ENCODER_REGISTRY.get('InstanceContextEncoder')(cfg, input_shape)

