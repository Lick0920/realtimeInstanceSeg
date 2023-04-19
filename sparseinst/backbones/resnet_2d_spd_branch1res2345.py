#
#  Copyright (c) Tianheng Cheng and its affiliates. All Rights Reserved
import torch
import math
import torch.nn as nn
from timm.models.resnet import BasicBlock, Bottleneck
from timm.models.layers import DropBlock2d, DropPath, AvgPool2dSame

from detectron2.layers import ShapeSpec, FrozenBatchNorm2d
from detectron2.modeling import Backbone, BACKBONE_REGISTRY
from detectron2.layers import NaiveSyncBatchNorm, DeformConv


def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class space_to_depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


class DeformableBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super().__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        # use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2_offset = nn.Conv2d(
            first_planes,
            18,
            kernel_size=3,
            stride=stride,
            padding=first_dilation,
            dilation=first_dilation
        )
        self.conv2 = DeformConv(
            first_planes,
            width,
            kernel_size=3,
            stride=stride,
            padding=first_dilation,
            bias=False,
            dilation=first_dilation,
        )

        self.bn2 = norm_layer(width)
        self.act2 = act_layer(inplace=True)
        # self.aa = aa_layer(channels=width, stride=stride) if use_aa else None

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        # self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        # self.drop_block = drop_block
        # self.drop_path = drop_path

        nn.init.constant_(self.conv2_offset.weight, 0)
        nn.init.constant_(self.conv2_offset.bias, 0)

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)

        x = self.act1(x)

        offset = self.conv2_offset(x)
        x = self.conv2(x, offset)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)

        return x


BLOCK_TYPE = {
    "basic": BasicBlock,
    "bottleneck": Bottleneck,
    "deform_bottleneck": DeformableBottleneck
}


def downsample_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False),
        norm_layer(out_channels)
    ])


def downsample_avg(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(*[
        pool,
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        norm_layer(out_channels)
    ])


def drop_blocks(drop_block_rate=0.):
    return [
        None, None,
        DropBlock2d(drop_block_rate, 5, 0.25) if drop_block_rate else None,
        DropBlock2d(drop_block_rate, 3, 1.00) if drop_block_rate else None]


def make_blocks(
        stage_block, channels, block_repeats, inplanes, reduce_first=1, output_stride=32,
        down_kernel_size=1, avg_down=False, drop_block_rate=0., drop_path_rate=0., **kwargs):
    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    net_stride = 4
    dilation = prev_dilation = 1
    for stage_idx, (planes, num_blocks, db) in enumerate(zip(channels, block_repeats, drop_blocks(drop_block_rate))):
        # choose block_fn through the BLOCK_TYPE
        block_fn = BLOCK_TYPE[stage_block[stage_idx]]

        stage_name = f'layer{stage_idx + 1}'  # never liked this name, but weight compat requires it
        stride = 1 if stage_idx == 0 else 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes, out_channels=planes * block_fn.expansion, kernel_size=down_kernel_size,
                stride=stride, dilation=dilation, first_dilation=prev_dilation, norm_layer=kwargs.get('norm_layer'))
            downsample = downsample_avg(
                **down_kwargs) if avg_down else downsample_conv(**down_kwargs)

        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, drop_block=db, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            block_dpr = drop_path_rate * net_block_idx / \
                (net_num_blocks - 1)  # stochastic depth linear decay rule
            blocks.append(block_fn(
                inplanes, planes, stride, downsample, first_dilation=prev_dilation,
                drop_path=DropPath(block_dpr) if block_dpr > 0. else None, **block_kwargs))
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))

    return stages, feature_info


class ResNet(Backbone):
    def __init__(self, block_types, layers, in_chans=3,
                 cardinality=1, base_width=64, stem_width=64, stem_type='', replace_stem_pool=False,
                 output_stride=32, block_reduce_first=1, down_kernel_size=1, avg_down=False,
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, aa_layer=None, drop_rate=0.0, drop_path_rate=0.,
                 drop_block_rate=0., global_pool='avg', zero_init_last_bn=True, block_args=None, out_features=None):
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        # self.num_classes = num_classes
        self.drop_rate = drop_rate
        super(ResNet, self).__init__()

        # Stem
        deep_stem = 'deep' in stem_type
        inplanes = stem_width * 2 if deep_stem else 64
        if deep_stem:
            stem_chs = (stem_width, stem_width)
            if 'tiered' in stem_type:
                stem_chs = (3 * (stem_width // 4), stem_width)
            self.conv1 = nn.Sequential(*[
                nn.Conv2d(in_chans, stem_chs[0], 3, stride=2, padding=1, bias=False),
                norm_layer(stem_chs[0]),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
                norm_layer(stem_chs[1]),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs[1], inplanes, 3, stride=1, padding=1, bias=False)])
        else:
            self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7,
                                   stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(inplanes)
        self.act1 = act_layer(inplace=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]

        # Stem Pooling
        if replace_stem_pool:
            self.maxpool = nn.Sequential(*filter(None, [
                nn.Conv2d(inplanes, inplanes, 3, stride=1 if aa_layer else 2, padding=1, bias=False),
                aa_layer(channels=inplanes, stride=2) if aa_layer else None,
                norm_layer(inplanes),
                act_layer(inplace=True)
            ]))
        else:
            if aa_layer is not None:
                self.maxpool = nn.Sequential(*[
                    nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                    aa_layer(channels=inplanes, stride=2)])
            else:
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Feature Blocks
        channels = [64, 128, 256, 512]
        stage_modules, stage_feature_info = make_blocks(
            block_types, channels, layers, inplanes, cardinality=cardinality, base_width=base_width,
            output_stride=output_stride, reduce_first=block_reduce_first, avg_down=avg_down,
            down_kernel_size=down_kernel_size, act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer,
            drop_block_rate=drop_block_rate, drop_path_rate=drop_path_rate, **block_args)
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        for n, m in self.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
        if zero_init_last_bn:
            for m in self.modules():
                if hasattr(m, 'zero_init_last_bn'):
                    m.zero_init_last_bn()

        ############ output_name 和 output_channel 要提前定义 ###########################
        out_features_names = ["spd_branch","res2", "res3", "res4", "res5"]
        self._out_feature_strides = dict(zip(out_features_names, [2, 4, 8, 16, 32])) # stride不知道什么时候用到的
        self._out_feature_channels = dict(
            zip(out_features_names, [x * BLOCK_TYPE[block_types[0]].expansion for x in [192, 64, 128, 256, 512]]))
        if out_features is None:
            self._out_features = out_features_names
        else:
            self._out_features = out_features
        
        # 字典添加一个 lck 0414 ---> 用于尝试增加一路输出 但是decoder分辨率不变，有没有提升
        self._out_feature_channels["spd_branch"] = 768 # /4
        if out_features is None:
            self._out_features = out_features_names
        else:
            self._out_features = out_features
        spd_channels_list = [3, 48, 192]
        # spd_channels_list = [64, 256, 1024]
        # channels 3 --12--> 48 --192--> 768 
        all_branches = []
        for i in range(2): # 
            out_channels = spd_channels_list[i]
            layer_branchspd = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, stride=1, kernel_size=3, padding=1, bias=False),
                space_to_depth(),
                nn.BatchNorm2d(4*out_channels), # 12
                nn.ReLU(inplace=True),

                nn.Conv2d(4*out_channels, out_channels*4*4, kernel_size=1, bias=False), # 12 --> 48
                nn.BatchNorm2d(out_channels*4*4)                     
            )
            all_branches.append(layer_branchspd)

        self.spd_branch = nn.Sequential(*all_branches)

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def size_divisibility(self):
        return 32

    def forward(self, x):
        print("input shape", x.shape)
        outputs = {}
        x_spd = x.clone()
        print("before spd branch shape", x_spd.size())
        # for i in range(3):
        #     x_spd = torch.cat([x_spd[..., ::2, ::2], x_spd[..., 1::2, ::2], x_spd[..., ::2, 1::2], x_spd[..., 1::2, 1::2]], 1)
        x_spd = self.spd_branch(x_spd)
        outputs["spd_branch"] = x_spd
        print("spd_branch", outputs["spd_branch"].size())
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        print("after conv1 maxpooling shape", x.shape)
        x = self.layer1(x)
        # 如果outputs["res2"] 的分辨率更大一些？ 不用这个而是用2d SPD分支得到的是不是好一些？
        outputs["res2"] = x
        print("res2 shape", x.shape)
        x = self.layer2(x)
        outputs["res3"] = x
        x = self.layer3(x)
        outputs["res4"] = x
        x = self.layer4(x)
        outputs["res5"] = x
        return outputs


@BACKBONE_REGISTRY.register()
def build_resnet_vd_spd_branch_backbone(cfg, input_shape):

    depth = cfg.MODEL.RESNETS.DEPTH
    norm_name = cfg.MODEL.RESNETS.NORM
    if norm_name == "FrozenBN":
        norm = FrozenBatchNorm2d
    elif norm_name == "SyncBN":
        norm = NaiveSyncBatchNorm
    else:
        norm = nn.BatchNorm2d
    if depth == 50:
        layers = [3, 4, 6, 3]
    elif depth == 101:
        layers = [3, 4, 23, 3]
    else:
        raise NotImplementedError()

    stage_blocks = []
    use_deformable = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    for idx in range(4):
        if use_deformable[idx]:
            stage_blocks.append("deform_bottleneck")
        else:
            stage_blocks.append("bottleneck")

    model = ResNet(stage_blocks, layers, stem_type="deep",
                   stem_width=32, avg_down=True, norm_layer=norm)
    return model



if __name__ == '__main__':
    import torch
    from torchsummary import summary
    
    from detectron2.config import get_cfg
    from detectron2.engine import default_argument_parser, default_setup
    from detectron2.modeling import build_backbone
    import sys
    sys.path.append('D:\project_python\SparseInst\sparseinst/')
    from config import add_sparse_inst_config
    # from config import cfg
    def setup(args):
        """
        Create configs and perform basic setups.
        """
        cfg = get_cfg()
        add_sparse_inst_config(cfg)
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        cfg.freeze()
        default_setup(cfg, args)
        return cfg
    args = default_argument_parser()
    args.add_argument("--fp16", action="store_true",
                      help="support fp16 for inference")
    args = args.parse_args()

    # args.config_file = 'D:\project_python\SparseInst\configs/sparse_inst_r50vd_dcn_giam_aug.yaml'
    args.config_file = 'D:\project_python\SparseInst\configs\\fenzhi\\2d_fenzhi_spd1res2345_216x144.yaml'
    print("Command Line Args:", args)
    cfg = setup(args)
    model = build_backbone(cfg)
    model.eval()

    # 输入shape 112 112
    input = torch.randn(1, 3, 864, 576)
    output = model(input)
    # 打印shape
    for out in output:
        print(out, output[out].shape)

    
