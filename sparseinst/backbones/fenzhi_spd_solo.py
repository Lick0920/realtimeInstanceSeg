"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""
from detectron2.modeling import Backbone, BACKBONE_REGISTRY
import torch
import torch.nn as nn


class space_to_depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        layers = [
                  nn.Conv2d(in_channels, out_channels, kernel_size=1, bias = False),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=True),
        ]

        if stride ==2:

            layers2 = [
            nn.Conv2d(out_channels, out_channels,stride= 1, kernel_size=3, padding=1, bias= False),
            space_to_depth(),   # the output of this will result in 4*out_channels
            nn.BatchNorm2d(4*out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(4*out_channels, out_channels* BottleNeck.expansion, kernel_size=1, bias = False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),                       
            ]

        else:

            layers2 = [
            nn.Conv2d(out_channels, out_channels,stride= stride, kernel_size=3, padding=1, bias= False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels* BottleNeck.expansion, kernel_size=1, bias = False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),                       
            ]

        layers.extend(layers2)

        self.residual_function = torch.nn.Sequential(*layers)

		
        # self.residual_function = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(out_channels, out_channels, stride=1, kernel_size=3, padding=1, bias=False),
		# 	space_to_depth(),   # the output of this will result in 4*out_channels
        #     nn.BatchNorm2d(4*out_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(4*out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        # )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(Backbone):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True))

        self.conv1 = Focus(3, 64, k=1,s=1)


		
        #we use a different inputsize than the original paper ---> inputsize 
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 2)    # Here in_channels = 64, and num_block[0] = 64 and s = 1 
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        out_features_names = ["res2", "res3", "res4", "res5"]
        self._out_feature_strides = dict(zip(out_features_names, [4, 8, 16, 32]))
        self._out_feature_channels = dict(
            zip(out_features_names, [x * 4 for x in [64, 128, 256, 512]]))
        # if out_features is None:
        self._out_features = out_features_names
        # else:
        #     self._out_features = out_features

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        print("input",x.shape)
        output = self.conv1(x)
        print("conv1",output.shape)
        outputs = {}
        output = self.conv2_x(output)
        print("conv2",output.shape)
        # outputs["res2"] = output
        output = self.conv3_x(output)
        print("conv3",output.shape)
        outputs["res3"] = output
        output = self.conv4_x(output)
        print("conv4",output.shape)
        outputs["res4"] = output
        output = self.conv5_x(output)
        print("conv5",output.shape)
        outputs["res5"] = output
        # output = self.avg_pool(output)
        # output = output.view(output.size(0), -1)
        # output = self.fc(output)

        return outputs


def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])




@BACKBONE_REGISTRY.register()
def build_resnet_vd_backbone_spd_solo(cfg, input_shape):

    model = ResNet(BottleNeck, [3, 4, 6, 3])
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
    args.config_file = 'D:\project_python\SparseInst\configs\\fenzhi\\spd_solo.yaml'
    print("Command Line Args:", args)
    cfg = setup(args)
    model = build_backbone(cfg)
    model.eval()

    # 输入shape 112 112
    input = torch.randn(1, 3, 3072, 2048)
    output = model(input)
    for k,v in output.items():
        print(k,v.shape)
    # print(model)
    # print(output.shape)

    