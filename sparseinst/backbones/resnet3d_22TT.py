#
#  Copyright (c) Tianheng Cheng and its affiliates. All Rights Reserved

import math
import torch
import torch.nn as nn
from timm.models.resnet import BasicBlock
from timm.models.layers import DropBlock2d, DropPath, AvgPool2dSame, create_attn, get_attn

from detectron2.layers import ShapeSpec, FrozenBatchNorm2d
from detectron2.modeling import Backbone, BACKBONE_REGISTRY
from detectron2.layers import NaiveSyncBatchNorm, DeformConv


def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


"""
inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 attn_layer=None, aa_layer=None, drop_block=None, drop_path=None
"""


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

class Bottleneck(nn.Module):
    expansion = 4
    

    def create_aa(self, aa_layer, channels, stride=2, enable=True):
        if not aa_layer or not enable:
            return nn.Identity()
        return aa_layer(stride) if issubclass(aa_layer, nn.AvgPool2d) else aa_layer(channels=channels, stride=stride)

    def __init__(
            self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
            reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
            attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(width)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act2 = act_layer(inplace=True)
        self.aa = self.create_aa(aa_layer, channels=width, stride=stride, enable=use_aa)

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop_block(x)
        x = self.act2(x)
        x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)

        return x


class Bottleneck_3d(nn.Module):
    expansion = 4
    

    def create_aa(self, aa_layer, channels, stride=2, enable=True):
        if not aa_layer or not enable:
            return nn.Identity()
        return aa_layer(stride) if issubclass(aa_layer, nn.AvgPool2d) else aa_layer(channels=channels, stride=stride)

    def __init__(
            self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
            reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
            attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(Bottleneck_3d, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        # self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        # self.bn1 = norm_layer(first_planes)
        # self.act1 = act_layer(inplace=True)
        self.conv1 = nn.Conv3d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(first_planes)
        self.act1 = act_layer(inplace=True)

        # self.conv2 = nn.Conv2d(
        #     first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
        #     padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        # self.bn2 = norm_layer(width)
        # self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        # self.act2 = act_layer(inplace=True)
        self.conv2 = nn.Conv3d(
            first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm3d(width)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act2 = act_layer(inplace=True)
        self.aa = self.create_aa(aa_layer, channels=width, stride=stride, enable=use_aa)

        # self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        # self.bn3 = norm_layer(outplanes)

        self.conv3 = nn.Conv3d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x): # 1 64 4 64 64
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop_block(x)
        x = self.act2(x)
        x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)

        return x

BLOCK_TYPE = {
    "basic": BasicBlock,
    "bottleneck": Bottleneck,
    "deform_bottleneck": DeformableBottleneck,
    "bottleneck_3d": Bottleneck_3d
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


def downsample_conv_3d(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False),
        norm_layer(out_channels)
    ])


def downsample_avg_3d( 
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = nn.BatchNorm3d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool3d  # AvgPool2dSame ---》 3d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(*[
        pool,
        nn.Conv3d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        norm_layer(out_channels)
    ])

def downsample_avg_3d_notdown( 
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = nn.BatchNorm3d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool3d  # AvgPool2dSame ---》 3d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(*[
        pool,
        nn.Conv3d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
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
            if stage_idx == 0 or stage_idx == 1:
                # 3d downsample
                downsample = downsample_avg_3d(
                    **down_kwargs) if avg_down else downsample_conv_3d(**down_kwargs)
            # elif stage_idx == 1:
            #     downsample = downsample_avg_3d_notdown(
            #         **down_kwargs) if avg_down else downsample_conv_3d(**down_kwargs)
            else:
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
        
        # init 3d conv 参数
        self.patch_slice = 8
        # 通道混洗的 分组数
        self.num_groups = 2

        # Stem
        deep_stem = 'deep' in stem_type
        inplanes = stem_width * 2 if deep_stem else 64
        if deep_stem:
            stem_chs = (stem_width, stem_width)
            if 'tiered' in stem_type:
                stem_chs = (3 * (stem_width // 4), stem_width)
            self.conv1 = nn.Sequential(*[
                nn.Conv3d(in_chans, stem_chs[0], 3, stride=(1,2,2), padding=1, bias=False),
                nn.BatchNorm3d(stem_chs[0]),
                act_layer(inplace=True),
                nn.Conv3d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(stem_chs[1]),
                act_layer(inplace=True),
                nn.Conv3d(stem_chs[1], inplanes, 3, stride=1, padding=1, bias=False)])

        self.bn1 = nn.BatchNorm3d(inplanes)
        self.act1 = act_layer(inplace=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]

        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=(1,2,2), padding=1)

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

        out_features_names = ["res2", "res3", "res4", "res5"]
        self._out_feature_strides = dict(zip(out_features_names, [4, 8, 16, 32]))
        self._out_feature_channels = dict(
            zip(out_features_names, [x * BLOCK_TYPE[block_types[0]].expansion for x in [64, 128, 256, 512]]))
        if out_features is None:
            self._out_features = out_features_names
        else:
            self._out_features = out_features

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
        # x [1, 3, 512, 512]
        input_shape = x.size()
        print("input size",input_shape)

        input_shape_reshaped_list = x.clone().view_as(torch.zeros(input_shape[0], input_shape[1], self.patch_slice*self.patch_slice, input_shape[2]//self.patch_slice,input_shape[3]//self.patch_slice))
        for i in range(self.patch_slice):
            for j in range(self.patch_slice):
                input_shape_reshaped_list[:, :, i*self.patch_slice+j, :, :] = x[:, :, i::self.patch_slice, j::self.patch_slice]
        # # 可视化检查
        x = self.conv1(input_shape_reshaped_list)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x) # 3d maxpooling 1 64 1 64 64 # 下采样太多？# 立体维度不下采样 keep 4
        outputs = {}
        x = self.layer1(x)  # 1 256 4 64 64

        print("after layer1", x.size())
        
        
        x = self.layer2(x)
        # reshape 
        x_layer1_shape = x.size()
        x_layer1_reshape2d = x.clone().view_as(torch.zeros(x_layer1_shape[0],x_layer1_shape[1],x_layer1_shape[-2]*self.patch_slice//2,x_layer1_shape[-1]*self.patch_slice)) 
        for i in range(self.patch_slice//2):
            for j in range(self.patch_slice):
                x_layer1_reshape2d[:, :, i::self.patch_slice//2, j::self.patch_slice] = x[:, :, i*self.patch_slice//2+j, :, :].clone()

        # print("outputs[res3] ",x.shape)
        # torch.cuda.empty_cache()
        outputs["res3"] = x_layer1_reshape2d
        x = self.layer3(x_layer1_reshape2d)
        # print("outputs[res4] ",x.shape)
        outputs["res4"] = x
        x = self.layer4(x)
        # print("outputs[res5] ",x.shape)
        outputs["res5"] = x
        return outputs


@BACKBONE_REGISTRY.register()
def build_resnet_vd3d_22TT_backbone(cfg, input_shape):

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
        if use_deformable[idx] == True:
            stage_blocks.append("deform_bottleneck")
        elif use_deformable[idx] == 2:
            stage_blocks.append("bottleneck_3d")
        else:
            stage_blocks.append("bottleneck")

    model = ResNet(stage_blocks, layers, stem_type="deep",     # 两个bottleneck 两个deformable_bottleneck
                   stem_width=32, avg_down=True, norm_layer=norm)
                    #  block_args={'attn_layer': 'se'})

    return model


if __name__ == '__main__':
    import torch
    from torchsummary import summary
    
    # 随机生成一个输入tensor 1 3 12 12
    # 输入一张图像，read为tensor格式
    from PIL import Image
    import torchvision.transforms as transforms

    # 读取图片
    # img = Image.open("D:\\project_python\\CascadePSP\\demo\\big_sample_image.jpg")
    img = Image.open("E:\\dataset\\coco\\train2017\\000000000074.jpg")
    # 定义transforms，将图片转换为tensor格式
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((384,600))
    ])

    # 将图片转换为tensor格式
    img_tensor = transform(img)

    # 打印tensor的形状和数据类型
    print(img_tensor.shape)
    print(img_tensor.dtype)

    x = img_tensor
    input_shape = img_tensor.size()
    patch_slice = 4


    # # 定义需要重新排列的行的索引
    # # 生成数字列表
    # index = [i + j*patch_slice*patch_slice for i in range(0, patch_slice*patch_slice) for j in range(0, 384//patch_slice//patch_slice)]

    # # 输出结果
    # print(index)
    # # 将需要重新排列的行按照索引进行选择
    # selected_rows = torch.index_select(x, 1, torch.LongTensor(index))
    # # 把input_shape_reshaped_list转化成图像格式显示出来
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(1, 1)
    # print(selected_rows.shape)
    # img_np = selected_rows.numpy()
    # img_np = img_np.transpose((1, 2, 0))
    # axs.imshow(img_np)
    # plt.imshow(img_np)
    # plt.show()
   
    
    input_shape_reshaped_list = torch.zeros(input_shape[0],patch_slice*patch_slice, input_shape[1]//patch_slice,input_shape[2]//patch_slice)
    for i in range(patch_slice):
        for j in range(patch_slice):
            # 降采样方式
            input_shape_reshaped_list[:, i*patch_slice+j, :, :] = x[:, i::patch_slice, j::patch_slice]

    # input_shape_reshaped_list = torch.zeros(input_shape[0],patch_slice*patch_slice, input_shape[1]//patch_slice,input_shape[2]//patch_slice)
    # for i in range(patch_slice):
    #     for j in range(patch_slice):
    #         # input_shape_reshaped_list[:, i*patch_slice+j, :, :] = x[:, i*input_shape[1]//patch_slice:(i+1)*input_shape[1]//patch_slice, j*input_shape[2]//patch_slice:(j+1)*input_shape[2]//patch_slice]
    #         # 为了更好的信息交互，我要采用降采样的方式来reshape --> 15 26 37 48 行列重排
    #         # input_shape_reshaped_list[:, i*patch_slice+j, :, :] = x[:, i*input_shape[1]//patch_slice:(i+1)*input_shape[1]//patch_slice, j*input_shape[2]//patch_slice:(j+1)*input_shape[2]//patch_slice]
    #         input_shape_reshaped_list[:, i*patch_slice+j, :, :] = x[]
    # # 将x y 行列重排 
    
    # input_shape_reshaped_list = x.view(input_shape[0],patch_slice*patch_slice, input_shape[1]//patch_slice,input_shape[2]//patch_slice)

    # # 将第2个维度进行通道混洗
    # batch_size, num_channels, height, width = input_shape_reshaped_list.shape
    # num_groups = 2  # 将通道分成4组
    # channels_per_group = num_channels // num_groups
    # input_shape_reshaped_list = input_shape_reshaped_list.view(batch_size, num_groups, channels_per_group, height, width)
    # input_shape_reshaped_list = input_shape_reshaped_list.permute(0, 2, 1, 3, 4).contiguous()
    # input_shape_reshaped_list = input_shape_reshaped_list.view(batch_size, num_channels, height, width)

    # reshape_img = img_tensor.view(3, 16, img_tensor.shape[1]//4, img_tensor.shape[2]//4)
    res = []
    for i in range(16):
        img_tensor_p = input_shape_reshaped_list[:,i,::]
        res.append(img_tensor_p)
    
    # 转化成图像格式显示出来
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4, 4)
    for i,image_tensor in enumerate(res):
        print(image_tensor.shape)
        img_np = image_tensor.numpy()
        img_np = img_np.transpose((1, 2, 0))
        axs[i//4, i%4].imshow(img_np)
        plt.imshow(img_np)
    plt.show()
    
    # # reshape回去
    # # 将通道混洗后的张量reshape回去
    # input_shape_reshaped_list = input_shape_reshaped_list.view(batch_size, num_groups, channels_per_group, height, width)
    # input_shape_reshaped_list = input_shape_reshaped_list.permute(0, 2, 1, 3, 4).contiguous()
    # input_shape_reshaped_list = input_shape_reshaped_list.view(batch_size, num_channels, height, width)

    # output = torch.zeros(input_shape[0],input_shape[1], input_shape[2])
    # # 把切成16块的input_shape_reshaped_list拼接回去
    # for i in range(patch_slice):
    #     for j in range(patch_slice):
    #         # output[:, i*input_shape[1]//patch_slice:(i+1)*input_shape[1]//patch_slice, j*input_shape[2]//patch_slice:(j+1)*input_shape[2]//patch_slice] = input_shape_reshaped_list[:, i*patch_slice+j, :, :]
    #         # 行列交换
    #         # output[:, j*input_shape[2]//patch_slice:(j+1)*input_shape[2]//patch_slice, i*input_shape[1]//patch_slice:(i+1)*input_shape[1]//patch_slice] = input_shape_reshaped_list[:, i*patch_slice+j, :, :]
    #         # input_shape_reshaped_list = input_shape_reshaped_list.transpose_(2,3)
    #         output[:, i*input_shape[1]//patch_slice:(i+1)*input_shape[1]//patch_slice, j*input_shape[2]//patch_slice:(j+1)*input_shape[2]//patch_slice] = input_shape_reshaped_list[:, j*patch_slice+i, :, :]
    

    # 把降采样后的reshape回去
    output = torch.zeros(input_shape[0],input_shape[1], input_shape[2])
    for i in range(patch_slice):
        for j in range(patch_slice):
            output[:, i::patch_slice, j::patch_slice] = input_shape_reshaped_list[:, i*patch_slice+j, :, :]
            


     
    # 把input_shape_reshaped_list转化成图像格式显示出来
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 1)
    print(output.shape)
    img_np = output.numpy()
    img_np = img_np.transpose((1, 2, 0))
    axs.imshow(img_np)
    plt.imshow(img_np)
    plt.show()

    # input_tensor = torch.randn(1, 3, 256, 256)
    # # 将input分成16张小图，每张图的大小为64*64，第一张图是第一列的像素点，第二张图是第的像素点，以此类推
    # reshape = input_tensor.view(1, 3, 16, 64, 64)
    # print(reshape.size())
    # print(input_tensor)
    # print(reshape)

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

    args.config_file = 'D:\project_python\SparseInst\\configs\\3d_test\\resnet3d_22TT_l12.yaml'
    print("Command Line Args:", args)
    cfg = setup(args)
    model = build_backbone(cfg)
    model.eval()
    # 放到cuda
    device = torch.device('cuda:0')
    # model.to(device)
    # summary(model, (3, 512, 512))
    
    x = torch.randn(1, 3, 512, 512)
    out = model(x)
    print(out['res3'].shape,out['res4'].shape,out['res5'].shape)
    from mmcv.cnn.utils import get_model_complexity_info
    flops, params = get_model_complexity_info(model, (3, 384, 600))
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {(3, 384, 600)}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')
    # torch.save(model.state_dict(), 'resnet50_vd.pth')
