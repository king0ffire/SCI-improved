import torch
import torch.nn as nn
from loss import LossFunction


class EnhanceNetwork(nn.Module):
    def __init__(self, layers, channels):  # channel是中间channel，起始终止都是3
        super(EnhanceNetwork, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation  # padding=1

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(3),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)

        illu = fea + input
        illu = torch.clamp(illu, 0.0001, 1)

        return illu




class CalibrateRDB(nn.Module):
    def __init__(self, layers=2, channels=3, groups=2, rdbtimes=2):
        super(CalibrateRDB, self).__init__()
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        self.layers = layers
        self.rdbtimes = rdbtimes
        list = []
        self.inconv = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1,
                                              dilation=dilation, padding=padding),
                                    #nn.BatchNorm2d(channels),
                                    nn.ReLU())
        midchannel = channels
        growrate = channels
        for i in range(layers):
            list.append(RDBlock(midchannel, growrate, groups))
            midchannel += growrate
        self.RDB = nn.Sequential(*list)
        self.onexone = nn.Conv2d(midchannel, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.outconv = nn.Sequential(nn.Conv2d(3, 3, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
                                     nn.BatchNorm2d(3),
                                     nn.Sigmoid())

    def forward(self, input):
        conv = input
        for i in range(self.rdbtimes):  # 吃配置
            conv = self.inconv(conv)
            conv = self.RDB(conv)
            conv = self.onexone(conv)+input
        conv = self.outconv(conv)
        return conv-input


class Network(nn.Module):

    def __init__(self, stage=3):
        super(Network, self).__init__()
        self.stage = stage
        self.enhance = EnhanceNetwork(layers=1, channels=3)
        self.calibrate = CalibrateRDB(layers=2, channels=16, groups=1, rdbtimes=1)  # 图像超分？
        self._criterion = LossFunction()

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            #nn.init.xavier_normal(m.weight.data)
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):

        ilist, rlist, inlist, attlist = [], [], [], []
        input_op = input
        for i in range(self.stage):  # 两个模块
            inlist.append(input_op)
            i = self.enhance(input_op)  # 光照
            r = input / i  # 所需图像
            r = torch.clamp(r, 0, 1)  # 将数据夹到一个区间
            att = self.calibrate(r)  # 残差
            # att = torch.clamp(att,0,1)
            input_op = input + att  # 下一阶段的输入
            ilist.append(i)
            rlist.append(r)
            attlist.append(torch.abs(att))

        return ilist, rlist, inlist, attlist

    def _loss(self, input):
        i_list, en_list, in_list, _ = self(input)  # 调用forward
        loss = 0
        for i in range(self.stage):
            loss += self._criterion(in_list[i], i_list[i])  # 调用LossFunc网络
        return loss


class RDBlock(nn.Module):

    def __init__(self, channels, growrate, groups=1):  # 自增式增长
        super(RDBlock, self).__init__()
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        self.RDB = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=growrate, kernel_size=kernel_size, stride=1, padding=padding,
                      groups=groups),
            nn.BatchNorm2d(growrate),
            nn.ReLU())

    def forward(self, input):
        re = self.RDB(input)
        return torch.cat((input, re), 1)