#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@filename     : s_resnet.py
@description  : 这是SymNet 的版本

@time         : 2024/03/01/15
@author       : 🚀🔥 dny22
@Version      : 1.0
'''

import torch.nn as nn
import math
from .slimmable_ops import SwitchableBatchNorm2d, TwoInputSequential
from .slimmable_ops import SlimmableConv2d, SlimmableLinear
from utils.config import FLAGS
import torch
import torch.utils.model_zoo as model_zoo
from .s_dsbn import DomainSpecificBatchNorm2d
from torchvision.models import resnet50

print("Total s_gpus:", torch.cuda.device_count())
gpus = torch.cuda.device_count()


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


# * resnet的细粒度的 block，就是残差块
class Block(nn.Module):
    def __init__(self, inp, outp, stride):
        super(Block, self).__init__()
        assert stride in [1, 2]
        self.skip = False
        midp = [i // 4 for i in outp]  # 得到中间层的下采样的chanel 数
        layers = [
            SlimmableConv2d(inp, midp, 1, 1, 0, bias=False),
            DomainSpecificBatchNorm2d(num_features=midp, num_classes=2),  # ** 对不同的域，使用不同域的 BN
            nn.ReLU(inplace=True),  # ** relu 不计入参数
            SlimmableConv2d(midp, midp, 3, stride, 1, bias=False),
            DomainSpecificBatchNorm2d(num_features=midp, num_classes=2),
            nn.ReLU(inplace=True),
            SlimmableConv2d(midp, outp, 1, 1, 0, bias=False),
            DomainSpecificBatchNorm2d(num_features=outp, num_classes=2),
        ]
        self.body = TwoInputSequential(*layers)  # ** 因为relu 不计入，所以 body只有[0,1, 3,4, 6,7]

        self.residual_connection = stride == 1 and inp == outp  # ？ 在 stride=1 且 输入和输出的 channel 一样，也就是每个配置的中间的？
        if not self.residual_connection:  # * 否则，就是 shortcut
            self.shortcut = TwoInputSequential(
                SlimmableConv2d(inp, outp, 1, stride=stride, bias=False),
                DomainSpecificBatchNorm2d(num_features=outp, num_classes=2),
            )
        self.post_relu = nn.ReLU(inplace=True)

    def forward(self, x, dom=0):
        self.dom = dom
        global id
        #         if id>15:
        #             id-=16
        dev = torch.cuda.current_device()
        # print("depth in block:",depth)
        # print("Blk dev: ",dev)
        # print("Block {} Input: {}".format(id,x.shape))
        if FLAGS.depth == 50:
            # ** 对下面每一个 stage
            if id[dev] in range(0, (0 + depth[0])) or id[dev] in range(3, (3 + depth[1])) or id[dev] in range(7, (7 + depth[2])) or id[dev] in range(13, (13 + depth[3])):
                # print("Block {} In: {}".format(id,x.shape))
                self.skip = False
                if self.residual_connection:  # * 是残差连接，就是直接加
                    res = self.body(x)
                    res += x
                else:  # * 如果是有 shortcut，就需要加上
                    res = self.body(x)
                    res += self.shortcut(x)
                res = self.post_relu(res)
                # print("Block {} Out: {}".format(id,res.shape))
                id[dev] += 1  # * 更新device 的载荷？
                return res
            else:
                self.skip = True
                # print("Block {} is skipped".format(id))
                id[dev] += 1
                return x
        # elif FLAGS.depth == 152:
        #     if id[dev] in range(0, (0 + depth[0])) or id[dev] in range(3, (3 + depth[1])) or id[dev] in range(11, (11 + depth[2])) or id[dev] in range(47, (47 + depth[3])):
        #         # print("Block {} In: {}".format(id,x.shape))
        #         self.skip = False
        #         if self.residual_connection:
        #             res = self.body(x)
        #             res += x
        #         else:
        #             res = self.body(x)
        #             res += self.shortcut(x)
        #         res = self.post_relu(res)
        #         # print("Block {} Out: {}".format(id,res.shape))
        #         id[dev] += 1
        #         return res
        #     else:
        #         self.skip = True
        #         # print("Block {} is skipped".format(id))
        #         id[dev] += 1
        #         return x
        # elif FLAGS.depth == 101:
        #     if id[dev] in range(0, (0 + depth[0])) or id[dev] in range(3, (3 + depth[1])) or id[dev] in range(7, (7 + depth[2])) or id[dev] in range(30, (30 + depth[3])):
        #         # print("Block {} In: {}".format(id,x.shape))
        #         self.skip = False
        #         if self.residual_connection:
        #             res = self.body(x)
        #             res += x
        #         else:
        #             res = self.body(x)
        #             res += self.shortcut(x)
        #         res = self.post_relu(res)
        #         # print("Block {} Out: {}".format(id,res.shape))
        #         id[dev] += 1
        #         return res
        # else:
        #     self.skip = True
        #     # print("Block {} is skipped".format(id))
        #     id[dev] += 1
        #     return x
        else:
            raise NotImplementedError


# * 模型入口
class Model(nn.Module):
    def __init__(self, num_classes=1000, input_size=224):
        super(Model, self).__init__()

        self.features_ini = []
        self.features_blks = []
        self.features_last = []
        # head
        assert input_size % 32 == 0  # 是32的整数，5次压缩，224/32=5
        self.depth_mult = None

        # setting of inverted residual blocks
        self.block_setting_dict = {
            # : [stage1, stage2, stage3, stage4]
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }
        self.block_setting = self.block_setting_dict[FLAGS.depth]  # 得到 block 的设置
        feats = [64, 128, 256, 512]  # 特征维度
        channels = [int(64 * width_mult) for width_mult in FLAGS.width_mult_list]  # [0.9, 1] # 对 channel 进行缩放
        self.features_ini = TwoInputSequential(  # *有两种 channel， in_channels_list, out_channels_list, kernel_size, stride=1, padding=0, dilation=1, groups_list=[1], bias=True
            SlimmableConv2d([3 for _ in range(len(channels))], channels, 7, 2, 3, bias=False),  # in_channel_list=[3,3], out_channel_list=[64*width_mult]  # kernel_size=7, stride=2, padding=3,  # * 最初的输入层，所以，输入的 channel 都是3
            DomainSpecificBatchNorm2d(num_features=channels, num_classes=2),  # 专门的域
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),  # kernel_size=3, stride=2, padding=1
        )  # * 这里是最初的输入。

        # * 构建resnet backbone
        for stage_id, n in enumerate(self.block_setting):  # resnet的block
            outp = [int(feats[stage_id] * width_mult * 4) for width_mult in FLAGS.width_mult_list]  # 输出的特征的 channel  # 得到输出的 channel数
            for i in range(n):
                if i == 0 and stage_id != 0:  # * 如果是 shortcut
                    self.features_blks.append(Block(channels, outp, 2))
                else:
                    self.features_blks.append(Block(channels, outp, 1))  # Block(inp, outp, stride)

                channels = outp  # 更新

        avg_pool_size = input_size // 32  # 均值pool
        self.features_last.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))

        # make it nn.Sequential
        self.features_blks = TwoInputSequential(*self.features_blks)
        self.features_last = TwoInputSequential(*self.features_last)

        # classifier
        self.outp = channels
        self.classifier = TwoInputSequential(SlimmableLinear(self.outp, [num_classes for _ in range(len(self.outp))]))

        # if FLAGS.reset_parameters:  # 重置参数
        #     self.reset_parameters()


        ## 这个在外边调用就可以了
        # if FLAGS.load_imagenet_parameters:
        #     _update_initial_weights_anyda(model_zoo.load_url(model_urls['resnet50']), num_classes=31, num_domains=2)

    # ? dom是指哪个域么?输出是多个网络的输出么？
    def forward(self, x, dom=0, alpha=0.0):
        global depth
        global id
        self.dom = dom
        id = [0] * gpus  # * 定义gpu 数量，记录 每个gpu 运行block 的数量
        # print("depth_mult",self.depth_mult)
        self.block_setting = [int(val * self.depth_mult) for val in self.block_setting_dict[FLAGS.depth]]
        depth = self.block_setting
        # print("depth",depth)
        # print('org x:',x.shape)
        x = self.features_ini(x, dom=self.dom)  # [32, 64, 56, 56]
        # print("Model global id: ",id)
        # print('after ini x:',x.shape)
        x = self.features_blks(x, dom=self.dom)  # [32, 2048, 7, 7]，做了5次下采样
        # print('after blk x:',x.shape)
        # print('after blk x:',x.shape)
        x = self.features_last(x)  # 最大池化？
        # print('after aapool2d x:',x.shape)
        # print('after aapool2d x:',x.shape)
        last_dim = x.size()[1]  # if x.size()[1]>=1843 else 1843
        # print("last dim b4 classify:", last_dim)
        features = x.view(-1, last_dim)


        probs = self.classifier(features)
        return probs  # 得到概率[B, cls]

    # * 直接就重置参数了？
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def load_imagenet_weights(
        self,
    ):
        pretrained_model = resnet50(pretrained=True)

        bns = []
        convs = []
        for name, module in pretrained_model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                convs.append(module.weight)
            elif isinstance(module, torch.nn.BatchNorm2d):
                bns.append(module)

        conv_layer_id = 0
        bn_layer_id = 0
        for name, module in self.named_modules():
            if isinstance(module, SlimmableConv2d):
                # assert
                # print(module.weight.shape, convs[conv_layer_id].shape)
                module.weight = convs[conv_layer_id]
                conv_layer_id += 1

            if isinstance(module, DomainSpecificBatchNorm2d):

                for i in range(2):  # 2 domains
                    if isinstance(module.bns[i], SwitchableBatchNorm2d):
                        for domain_bn in module.bns[i].bn:

                            # print(domain_bn.weight.shape, bns[bn_layer_id].weight[:domain_bn.weight.shape[0]].shape)
                            domain_bn.weight = torch.nn.Parameter(bns[bn_layer_id].weight[: domain_bn.weight.shape[0]])
                            domain_bn.bias = torch.nn.Parameter(bns[bn_layer_id].bias[: domain_bn.bias.shape[0]])
                    # if isinstance(module.bns[i], USBatchNorm2d):
                    #     module.bns[i].weight = bns[bn_layer_id].weight
                    #     module.bns[i].bias = bns[bn_layer_id].bias

                bn_layer_id += 1
