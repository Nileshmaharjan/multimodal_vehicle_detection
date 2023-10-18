import torch.nn as nn
import math
import torch


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

# num_channels: total no of output channel
# input_channel: total no of input channel
class WDSR(nn.Module):
    def __init__(self, num_channels=3,input_channel=64, factor=4, width=16, depth=16, kernel_size=3, conv=default_conv):
        super(WDSR, self).__init__()

        n_resblock = depth
        n_feats = width
        kernel_size = kernel_size
        scale = factor
        act=nn.ReLU(inplace=True)


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
