import torch.nn as nn
import math
import torch


class WideActivation(nn.Module):
    def __init__(self, num_parameters=2):
        super(WideActivation, self).__init__()
        self.num_parameters = num_parameters
        self.alpha = nn.Parameter(torch.Tensor(num_parameters))
        self.beta = nn.Parameter(torch.Tensor(num_parameters))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.alpha)
        nn.init.zeros_(self.beta)

    def forward(self, x):
        return self.alpha.view(1, self.num_parameters, 1, 1) * x + self.beta.view(1, self.num_parameters, 1, 1)

def make_model(args, parent=False):
    return WDSR(args)

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, act=WideActivation(), res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        m = []
        m.append(
            nn.Conv2d(n_feat, n_feat,kernel_size, padding=kernel_size//2)
        )
        m.append(act)
        m.append(
            nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size//2)
        )

        self.body = nn.Sequential(*m)

    def forward(self,x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


# num_channels: total no of output channel
# input_channel: total no of input channel
class WDSR(nn.Module):
    def __init__(self, num_channels=3,input_channel=64, factor=4, width=64, depth=16, kernel_size=3, conv=default_conv):
        super(WDSR, self).__init__()

        n_resblock = depth
        n_feats = width
        kernel_size = kernel_size
        scale = factor
        act=WideActivation()

        # define head module
        m_head = [conv(input_channel, n_feats, kernel_size, padding=3//2)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=1.
            ) for _ in range(n_resblock)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, num_channels, kernel_size)
        ]





# class WDSR(nn.Module):
#     def __init__(self, num_channels=3,input_channel=64, factor=4, width=64, depth=16, kernel_size=3, conv=default_conv):
#         super(WDSR, self).__init__()
#
#         n_resblock = depth
#         n_feats = width
#         kernel_size = kernel_size
#         scale = factor
#         act = nn.ReLU()
#
#         # rgb_mean = (0.4488, 0.4371, 0.4040)
#         # rgb_std = (1.0, 1.0, 1.0)
#         # self.sub_mean = common.MeanShift(1.0, rgb_mean, rgb_std)
#
#         # define head module
#         m_head = [conv(input_channel, n_feats, kernel_size)]
#
#         # define body module
#         m_body = [
#             ResBlock(
#                 conv, n_feats, kernel_size, act=act, res_scale=1.
#             ) for _ in range(n_resblock)
#         ]
#         m_body.append(conv(n_feats, n_feats, kernel_size))
#
#         # define tail module
#         m_tail = [
#             Upsampler(conv, scale, n_feats, act=False),
#             conv(n_feats, num_channels, kernel_size)
#         ]
#
#         # self.add_mean = common.MeanShift(1.0, rgb_mean, rgb_std, 1)
#
#         self.head = nn.Sequential(*m_head)
#         self.body = nn.Sequential(*m_body)
#         self.tail = nn.Sequential(*m_tail)
#
#     def forward(self, x):
#         # x = self.sub_mean(x)
#         x = self.head(x)
#
#         res = self.body(x)
#         res += x
#
#         x = self.tail(res)
#         # x = self.add_mean(x)
#
#         return x
#
#     def load_state_dict(self, state_dict, strict=True):
#         own_state = self.state_dict()
#         for name, param in state_dict.items():
#             if name in own_state:
#                 if isinstance(param, nn.Parameter):
#                     param = param.data
#                 try:
#                     own_state[name].copy_(param)
#                 except Exception:
#                     if name.find('tail') == -1:
#                         raise RuntimeError('While copying the parameter named {}, '
#                                            'whose dimensions in the model are {} and '
#                                            'whose dimensions in the checkpoint are {}.'
#                                            .format(name, own_state[name].size(), param.size()))
#             elif strict:
#                 if name.find('tail') == -1:
#                     raise KeyError('unexpected key "{}" in state_dict'
#                                    .format(name))


class WDSR(nn.Module):
    def __init__(self, num_channels=3,input_channel=64, factor=4, width=64, depth=16, kernel_size=3, conv=default_conv):
        super(WDSR, self).__init__()

        n_resblock = depth
        n_feats = width
        kernel_size = kernel_size
        scale = factor
        act = nn.ReLU()

        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # rgb_std = (1.0, 1.0, 1.0)
        # self.sub_mean = common.MeanShift(1.0, rgb_mean, rgb_std)

        # define head module
        m_head = [conv(input_channel, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=1.
            ) for _ in range(n_resblock)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, num_channels, kernel_size)
        ]

        # self.add_mean = common.MeanShift(1.0, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        # x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))