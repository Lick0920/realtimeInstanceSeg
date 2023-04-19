# Copyright (c) Tianheng Cheng and its affiliates. All Rights Reserved

import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

from detectron2.utils.registry import Registry
from detectron2.layers import Conv2d
from sparseinst.encoder import SPARSE_INST_ENCODER_REGISTRY
import pickle
import numpy as np
SPARSE_INST_DECODER_REGISTRY = Registry("SPARSE_INST_DECODER")
SPARSE_INST_DECODER_REGISTRY.__doc__ = "registry for SparseInst decoder"
import scipy.sparse as sp

def  _make_stack_3x3_convs(num_convs, in_channels, out_channels):
    convs = []
    for _ in range(num_convs):
        convs.append(
            Conv2d(in_channels, out_channels, 3, padding=1))
        convs.append(nn.ReLU(True))
        in_channels = out_channels
    return nn.Sequential(*convs)


class InstanceBranch(nn.Module):

    def __init__(self, cfg, in_channels):
        super().__init__()
        # norm = cfg.MODEL.SPARSE_INST.DECODER.NORM
        dim = cfg.MODEL.SPARSE_INST.DECODER.INST.DIM  # 256
        num_convs = cfg.MODEL.SPARSE_INST.DECODER.INST.CONVS # 4
        num_masks = cfg.MODEL.SPARSE_INST.DECODER.NUM_MASKS # 100
        kernel_dim = cfg.MODEL.SPARSE_INST.DECODER.KERNEL_DIM # 256
        self.num_classes = cfg.MODEL.SPARSE_INST.DECODER.NUM_CLASSES # 80
 
        self.inst_convs = _make_stack_3x3_convs(num_convs, in_channels, dim)
        # iam prediction, a simple conv
        self.iam_conv = nn.Conv2d(dim, num_masks, 3, padding=1)

        # outputs
        self.cls_score = nn.Linear(dim, self.num_classes) # 256 80
        self.mask_kernel = nn.Linear(dim, kernel_dim)  # 256 128 # for mask head just linear
        self.objectness = nn.Linear(dim, 1) # 256 1

        self.prior_prob = 0.01
        self._init_weights()

    def _init_weights(self):
        for m in self.inst_convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for module in [self.iam_conv, self.cls_score]:
            init.constant_(module.bias, bias_value)
        init.normal_(self.iam_conv.weight, std=0.01)
        init.normal_(self.cls_score.weight, std=0.01)

        init.normal_(self.mask_kernel.weight, std=0.01)
        init.constant_(self.mask_kernel.bias, 0.0)

    def forward(self, features):
        # instance features (x4 convs)
        features = self.inst_convs(features)
        # predict instance activation maps
        iam = self.iam_conv(features)
        iam_prob = iam.sigmoid()

        B, N = iam_prob.shape[:2]
        C = features.size(1)
        # BxNxHxW -> BxNx(HW)
        iam_prob = iam_prob.view(B, N, -1)
        normalizer = iam_prob.sum(-1).clamp(min=1e-6)
        iam_prob = iam_prob / normalizer[:, :, None]
        # aggregate features: BxCxHxW -> Bx(HW)xC
        inst_features = torch.bmm(
            iam_prob, features.view(B, C, -1).permute(0, 2, 1))
        # predict classification & segmentation kernel & objectness
        pred_logits = self.cls_score(inst_features)
        pred_kernel = self.mask_kernel(inst_features)
        pred_scores = self.objectness(inst_features)
        return pred_logits, pred_kernel, pred_scores, iam


class MaskBranch(nn.Module):

    def __init__(self, cfg, in_channels):
        super().__init__()
        dim = cfg.MODEL.SPARSE_INST.DECODER.MASK.DIM # 256
        num_convs = cfg.MODEL.SPARSE_INST.DECODER.MASK.CONVS # 4
        kernel_dim = cfg.MODEL.SPARSE_INST.DECODER.KERNEL_DIM # 128
        self.mask_convs = _make_stack_3x3_convs(num_convs, in_channels, dim) # 256 256
        self.projection = nn.Conv2d(dim, kernel_dim, kernel_size=1) # 256 128
        self._init_weights()

    def _init_weights(self):
        for m in self.mask_convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
        c2_msra_fill(self.projection)

    def forward(self, features):
        # mask features (x4 convs)
        features = self.mask_convs(features) # 258 256
        return self.projection(features) # 256 128


@SPARSE_INST_DECODER_REGISTRY.register()
class BaseIAMDecoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        # add 2 for coordinates
        in_channels = cfg.MODEL.SPARSE_INST.ENCODER.NUM_CHANNELS + 2

        self.scale_factor = cfg.MODEL.SPARSE_INST.DECODER.SCALE_FACTOR
        self.output_iam = cfg.MODEL.SPARSE_INST.DECODER.OUTPUT_IAM

        self.inst_branch = InstanceBranch(cfg, in_channels)
        self.mask_branch = MaskBranch(cfg, in_channels)

        # ####### lck  0418 ################
        # with open('D:\\project_python\\SparseInst\\coco_adj.pkl', 'rb') as f:
        #     self.adj_matrix = pickle.load(f).astype(np.float32)
        #     # self.adj_matrix = np.float32(self.adj_matrix)
            
        #     # 假设已经创建好了coo_matrix类型的矩阵coo_matrix
        #     coo_matrix = sp.coo_matrix(self.adj_matrix)  # 转成coo格式
        #     i = torch.LongTensor([coo_matrix.row, coo_matrix.col])  # 索引
        #     v = torch.FloatTensor(coo_matrix.data)  # 数据
        #     shape = coo_matrix.shape  # 形状
        #     tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()  # 创建稀疏张量
        #     self.adj_matrix = nn.Parameter(tensor, requires_grad=False)

        # self.graph_weight_fc = nn.Linear(1025, 128)
        # self.relu = nn.ReLU(inplace=True)
        # # ######################lck 0418 ################
        # # dim = 1024
        # # self.num_classes = cfg.MODEL.SPARSE_INST.DECODER.NUM_CLASSES # 80
        # # kernel_dim = cfg.MODEL.SPARSE_INST.DECODER.KERNEL_DIM # 128
        # # self.cls_score = nn.Linear(dim, self.num_classes) # 256 80
        # # self.mask_kernel = nn.Linear(dim, kernel_dim)  # 256 128 # for mask head just linear
        # # self.objectness = nn.Linear(dim, 1)
        

    @torch.no_grad()
    def compute_coordinates_linspace(self, x):
        # linspace is not supported in ONNX
        h, w = x.size(2), x.size(3)
        y_loc = torch.linspace(-1, 1, h, device=x.device)
        x_loc = torch.linspace(-1, 1, w, device=x.device)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        locations = torch.cat([x_loc, y_loc], 1)
        return locations.to(x)

    @torch.no_grad()
    def compute_coordinates(self, x):
        h, w = x.size(2), x.size(3)
        y_loc = -1.0 + 2.0 * torch.arange(h, device=x.device) / (h - 1)
        x_loc = -1.0 + 2.0 * torch.arange(w, device=x.device) / (w - 1)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        locations = torch.cat([x_loc, y_loc], 1)
        return locations.to(x)

    def forward(self, features):
        coord_features = self.compute_coordinates(features)
        features = torch.cat([coord_features, features], dim=1) # 1 256+2 384 256
        pred_logits, pred_kernel, pred_scores, iam = self.inst_branch(features) # pred_kernel 1 100 128 干啥的
        
        mask_features = self.mask_branch(features) # 1 128 384 256

        N = pred_kernel.shape[1] # 100 增强后 B 200 128 
        # mask_features: BxCxHxW
        B, C, H, W = mask_features.shape # B 128 H W
        pred_masks = torch.bmm(pred_kernel, mask_features.view(
            B, C, H * W)).view(B, N, H, W)  # 1 100 384 256

        pred_masks = F.interpolate(
            pred_masks, scale_factor=self.scale_factor,
            mode='bilinear', align_corners=False)# 1 100 384x2 256x2  -----》 200

        output = {
            "pred_logits": pred_logits, # 1 100 80  ----> 200
            "pred_masks": pred_masks, # 1 100 384 256 ---> 200
            "pred_scores": pred_scores, # 1 100 80  ----> 200
        }

        if self.output_iam:
            iam = F.interpolate(iam, scale_factor=self.scale_factor,
                                mode='bilinear', align_corners=False)
            output['pred_iam'] = iam

        return output


class GroupInstanceBranch(nn.Module):

    def __init__(self, cfg, in_channels):
        super().__init__()
        dim = cfg.MODEL.SPARSE_INST.DECODER.INST.DIM # 256
        num_convs = cfg.MODEL.SPARSE_INST.DECODER.INST.CONVS
        num_masks = cfg.MODEL.SPARSE_INST.DECODER.NUM_MASKS
        kernel_dim = cfg.MODEL.SPARSE_INST.DECODER.KERNEL_DIM
        self.num_groups = cfg.MODEL.SPARSE_INST.DECODER.GROUPS
        self.num_classes = cfg.MODEL.SPARSE_INST.DECODER.NUM_CLASSES

        self.inst_convs = _make_stack_3x3_convs(num_convs, in_channels, dim) # 258 256
        # iam prediction, a group conv
        expand_dim = dim * self.num_groups # 1024
        self.iam_conv = nn.Conv2d(
            dim, num_masks * self.num_groups, 3, padding=1, groups=self.num_groups) # 256 400 分组卷积 4组
        # outputs
        self.fc = nn.Linear(expand_dim, expand_dim)  # 1024 1024

        self.cls_score = nn.Linear(expand_dim, self.num_classes) # 1024 80
        self.mask_kernel = nn.Linear(expand_dim, kernel_dim) # 1024 128
        self.objectness = nn.Linear(expand_dim, 1)

        self.prior_prob = 0.01
        self._init_weights()

        ############### lck  0418 ###################
        with open('D:\\project_python\\SparseInst\\coco_adj.pkl', 'rb') as f:
            self.adj_matrix = pickle.load(f).astype(np.float32)
            # self.adj_matrix = np.float32(self.adj_matrix)
            
            # 假设已经创建好了coo_matrix类型的矩阵coo_matrix
            coo_matrix = sp.coo_matrix(self.adj_matrix)  # 转成coo格式
            i = torch.LongTensor([coo_matrix.row, coo_matrix.col])  # 索引
            v = torch.FloatTensor(coo_matrix.data)  # 数据
            shape = coo_matrix.shape  # 形状
            tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()  # 创建稀疏张量
            self.adj_matrix = nn.Parameter(tensor, requires_grad=False)

        self.graph_weight_fc = nn.Linear(1025, 1024) # 
        self.relu = nn.ReLU(inplace=True)
        # ######################lck 0418 ################
        # dim = 1024
        self.num_classes = cfg.MODEL.SPARSE_INST.DECODER.NUM_CLASSES # 80
        kernel_dim = cfg.MODEL.SPARSE_INST.DECODER.KERNEL_DIM # 128
        self.cls_score_new = nn.Linear(expand_dim, self.num_classes) # 256 80
        self.mask_kernel_new = nn.Linear(expand_dim, kernel_dim)  # 256 128 # for mask head just linear
        self.objectness_new = nn.Linear(expand_dim, 1)

    def _init_weights(self):
        for m in self.inst_convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for module in [self.iam_conv, self.cls_score]:
            init.constant_(module.bias, bias_value)
        init.normal_(self.iam_conv.weight, std=0.01)
        init.normal_(self.cls_score.weight, std=0.01)

        init.normal_(self.mask_kernel.weight, std=0.01)
        init.constant_(self.mask_kernel.bias, 0.0)
        c2_xavier_fill(self.fc)

    def forward(self, features):
        # instance features (x4 convs)
        features = self.inst_convs(features) # B C H W
        # predict instance activation maps
        iam = self.iam_conv(features) # 1 400 H W 一个卷积
        iam_prob = iam.sigmoid() # 1 400 H W

        B, N = iam_prob.shape[:2]
        C = features.size(1)
        # BxNxHxW -> BxNx(HW)
        iam_prob = iam_prob.view(B, N, -1)
        normalizer = iam_prob.sum(-1).clamp(min=1e-6)
        iam_prob = iam_prob / normalizer[:, :, None]

        # aggregate features: BxCxHxW -> Bx(HW)xC
        inst_features = torch.bmm(
            iam_prob, features.view(B, C, -1).permute(0, 2, 1)) # 1 400 C
        # 分 group 1 100 Cx4
        inst_features = inst_features.reshape(
            B, 4, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1) # 1 100 Cx4

        inst_features = F.relu_(self.fc(inst_features))
        # predict classification & segmentation kernel & objectness
        pred_logits = self.cls_score(inst_features)  # 都是线性层 1024 80  weight 80 1024 bias 80  ---》 B 100 80
        pred_kernel = self.mask_kernel(inst_features) # 1024 128  ---》 去乘 mask的？  B 100 128
        pred_scores = self.objectness(inst_features) # 1024 1   B 100 1

        ############ 0418 #### lck ##############
        ############ 语义增强 pred_kernel 修改 #########
        # 语义增强特征
        # 1.build global semantic pool
        global_semantic_pool = torch.cat((self.cls_score.weight,
                                            self.cls_score.bias.unsqueeze(1)), 1).detach() # 80 1024
        # print("global semantic pool data",global_semantic_pool.data.max(), global_semantic_pool.data.min())
        # 2.compute graph attention  ---》 不要 attention_map
        # attention_map = nn.Softmax(1)(torch.mm(features, torch.transpose(global_semantic_pool, 0, 1)))
        # 3.adaptive global reasoning
        # alpha_em = attention_map.unsqueeze(-1) * torch.mm(self.adj_gt, global_semantic_pool).unsqueeze(0)
        # alpha_em = alpha_em.view(-1, global_semantic_pool.size(-1))
        ######### 自己理解的 不加attention ######
        # print("adj_matrix", self.adj_matrix.data.max(), self.adj_matrix.data.min())
        alpha_em = torch.mm(self.adj_matrix, global_semantic_pool)
        alpha_em = self.graph_weight_fc(alpha_em)
        alpha_em = self.relu(alpha_em) # 80 128
        # n_classes = self.inst_branch.cls_score.weight.size(0)
        cls_prob = nn.Softmax(2)(pred_logits) # 8 100 80
        # print("cls_prob", cls_prob.data.max(), cls_prob.data.min())
        enhenced_f = torch.bmm(cls_prob, torch.unsqueeze(alpha_em, dim=0).repeat(cls_prob.shape[0], 1,1)) # B 100 Eout
        # pred_kernel = enhenced_f   # 不能直接用
        # 按照自己理解修改 ##### 
        # cls_prob = nn.Softmax(1)(pred_scores).view(len(img_meta), -1, n_classes)
        # enhanced_feat = torch.bmm(cls_prob, alpha_em.view(len(img_meta), -1, self.graph_out_channels))
        # enhanced_feat = enhanced_feat.view(-1, self.graph_out_channels)

        # 还是再训练新的分类器 
        addcls_feature = torch.cat((inst_features, enhenced_f), dim=1)
        pred_logits = self.cls_score_new(addcls_feature)
        pred_kernel = self.mask_kernel_new(addcls_feature)
        pred_scores = self.objectness_new(addcls_feature)
        #######################################################################################

        return pred_logits, pred_kernel, pred_scores, iam


@SPARSE_INST_DECODER_REGISTRY.register()
class GroupIAMDecoder(BaseIAMDecoder):

    def __init__(self, cfg):
        super().__init__(cfg)
        in_channels = cfg.MODEL.SPARSE_INST.ENCODER.NUM_CHANNELS + 2
        self.inst_branch = GroupInstanceBranch(cfg, in_channels)


class GroupInstanceSoftBranch(GroupInstanceBranch):

    def __init__(self, cfg, in_channels):
        super().__init__(cfg, in_channels)
        self.softmax_bias = nn.Parameter(torch.ones([1, ]))

    def forward(self, features):
        # instance features (x4 convs)
        features = self.inst_convs(features)
        # predict instance activation maps
        iam = self.iam_conv(features)

        B, N = iam.shape[:2]
        C = features.size(1)
        # BxNxHxW -> BxNx(HW)
        iam_prob = F.softmax(iam.view(B, N, -1) + self.softmax_bias, dim=-1)
        # aggregate features: BxCxHxW -> Bx(HW)xC
        inst_features = torch.bmm(
            iam_prob, features.view(B, C, -1).permute(0, 2, 1))

        inst_features = inst_features.reshape(
            B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)

        inst_features = F.relu_(self.fc(inst_features))
        # predict classification & segmentation kernel & objectness
        pred_logits = self.cls_score(inst_features)
        pred_kernel = self.mask_kernel(inst_features)
        pred_scores = self.objectness(inst_features)
        return pred_logits, pred_kernel, pred_scores, iam


@SPARSE_INST_DECODER_REGISTRY.register()
class GroupIAMSoftDecoder(BaseIAMDecoder):

    def __init__(self, cfg):
        super().__init__(cfg)
        in_channels = cfg.MODEL.SPARSE_INST.ENCODER.NUM_CHANNELS + 2
        self.inst_branch = GroupInstanceSoftBranch(cfg, in_channels)


def build_sparse_inst_decoder(cfg):
    name = cfg.MODEL.SPARSE_INST.DECODER.NAME  # GroupIAMSoftDecoder
    return SPARSE_INST_DECODER_REGISTRY.get(name)(cfg)
