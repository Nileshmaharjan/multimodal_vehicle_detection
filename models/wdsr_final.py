import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


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
    def __init__(self, conv, n_feat, kernel_size, act=nn.ReLU(inplace=True), res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(n_feat, n_feat//4, 1)
        self.bn1 = nn.BatchNorm2d(n_feat//4)
        self.conv2 = nn.Conv2d(n_feat//4, n_feat//4, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(n_feat//4)
        self.conv3 = nn.Conv2d(n_feat//4, n_feat, 1)
        self.bn3 = nn.BatchNorm2d(n_feat)
        self.act = act

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out.mul(self.res_scale)
        out += identity
        out = self.act(out)

        return out

# # num_channels: total no of output channel
# # input_channel: total no of input channel
# class WDSR(nn.Module):
#     def __init__(self, num_channels=3,input_channel=64, factor=4, width=16, depth=16, kernel_size=3, conv=default_conv):
#         super(WDSR, self).__init__()
#
#         n_resblock = depth
#         n_feats = width
#         kernel_size = kernel_size
#         scale = factor
#         act=nn.ReLU(inplace=True)
#
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
#
#         m_tail = [
#             Upsampler(conv, scale, n_feats, act=False),
#             conv(n_feats, num_channels, kernel_size)
#         ]
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

class WDSR(nn.Module):
    def __init__(self, num_channels=3, input_channel=64, factor=4, width=16, depth=16, kernel_size=3, conv=default_conv):
        super(WDSR, self).__init__()

        n_resblock = depth
        n_feats = width
        kernel_size = kernel_size
        scale = factor
        act = nn.ReLU(inplace=True)

        # define head module
        m_head = [conv(input_channel, n_feats, kernel_size)]

        # Add the CBAM module here
        self.cbam = CBAM(n_feats, reduction_ratio=16, pool_types=['avg', 'max'])

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

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)

        # Apply CBAM module after the head
        x = self.cbam(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return x
