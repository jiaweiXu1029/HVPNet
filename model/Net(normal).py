import math
from model.smt import smt_t,smt_b,smt_l
from model.MobileNetV2 import mobilenet_v2
import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from MSRA import RAMOptimized

TRAIN_SIZE = 384
import os
import numpy as np
from thop import profile, clever_format

class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class CorrelationModule(nn.Module):
    def __init__(self, all_channel=64):
        super(CorrelationModule, self).__init__()
        self.linear_e = nn.Linear(all_channel, all_channel, bias=False)
        self.channel = all_channel
        self.fusion = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)

    def forward(self, exemplar, query):  # exemplar: middle, query: rgb or T
        fea_size = exemplar.size()[2:]
        all_dim = fea_size[0] * fea_size[1]
        exemplar_flat = exemplar.view(-1, self.channel, all_dim)  # N,C,H*W
        query_flat = query.view(-1, self.channel, all_dim)
        exemplar_t = torch.transpose(exemplar_flat, 1, 2).contiguous()  # batchsize x dim x num, N,H*W,C
        exemplar_corr = self.linear_e(exemplar_t)  #
        A = torch.bmm(exemplar_corr, query_flat)
        B = F.softmax(torch.transpose(A, 1, 2), dim=1)
        exemplar_att = torch.bmm(query_flat, B).contiguous()

        exemplar_att = exemplar_att.view(-1, self.channel, fea_size[0], fea_size[1])
        exemplar_out = self.fusion(exemplar_att)

        return exemplar_out

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up_sample = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                                       nn.BatchNorm2d(out_channels),
                                       nn.GELU(),
                                       nn.UpsamplingBilinear2d(scale_factor=2, ))

    def forward(self, inputs):
        return self.up_sample(inputs)

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            # out = F.normalize(out)
            out = F.relu(out, inplace=True)
            # out = self.dropout(out)
        return out

class CRF(torch.nn.Module):
    def __init__(self, channels):
        super(CRF, self).__init__()
        ws = [3, 3, 3, 3]
        self.conv_fusion = ConvLayer(2 * channels, channels, 3, 1)

        self.conv_sum = ConvLayer(channels, channels, 3, 1)
        self.conv_mul = ConvLayer(channels, channels, 3, 1)

        block = []
        block += [ConvLayer(2 * channels, channels, 1, 1),
                  ConvLayer(channels, channels, 3, 1),
                  ConvLayer(channels, channels, 3, 1)]
        self.bottelblock = nn.Sequential(*block)

    def forward(self, x_ir, x_vi):
        # initial fusion - conv
        # print('conv')
        f_sum = x_ir + x_vi
        f_mul = x_ir * x_vi
        f_init = torch.cat([f_sum, f_sum], 1)
        f_init = self.conv_fusion(f_init)

        out_ir = self.conv_sum(f_sum)
        out_vi = self.conv_mul(f_mul)
        out = torch.cat([out_ir, out_vi], 1)
        out = self.bottelblock(out)
        out = f_init + out
        return out

class CLM(nn.Module):
    def __init__(self, all_channel=64):
        super(CLM, self).__init__()
        self.corr_x_2_x_ir = CorrelationModule(all_channel)
        self.corr_ir_2_x_ir = CorrelationModule(all_channel)
        self.smooth1 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.smooth2 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.fusion = BasicConv2d(2 * all_channel, all_channel, kernel_size=3, padding=1)
        self.pred = nn.Conv2d(all_channel, 2, kernel_size=3, padding=1, bias=True)

    def forward(self, x, x_ir, ir):  # exemplar: middle, query: rgb or T
        corr_x_2_x_ir = self.corr_x_2_x_ir(x_ir, x)
        corr_ir_2_x_ir = self.corr_ir_2_x_ir(x_ir, ir)

        summation = self.smooth1(corr_x_2_x_ir + corr_ir_2_x_ir)
        multiplication = self.smooth2(corr_x_2_x_ir * corr_ir_2_x_ir)

        fusion = self.fusion(torch.cat([summation, multiplication], 1))
        sal_pred = self.pred(fusion)

        return fusion, sal_pred

from SRA import SRA  # 假设已经正确导入SRA模块

class SCA(nn.Module):
    def __init__(self, all_channel=64, head_num=4, window_size=7):
        super(SCA, self).__init__()
        self.sra = SRA(dim=all_channel, head_num=head_num, window_size=window_size)


        self.conv2 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)

    def forward(self, x, ir):
        multiplication = x * ir
        summation = self.conv2(x + ir)

        sa = self.sra(multiplication)
        summation_sa = summation.mul(sa)

        sc_feat = self.sra(summation_sa)

        return sc_feat

class EDS(nn.Module):
    def __init__(self, all_channel=64):
        super(EDS, self).__init__()
        self.conv1 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.dconv1 = BasicConv2d(all_channel, int(all_channel / 4), kernel_size=3, padding=1)
        self.dconv2 = BasicConv2d(all_channel, int(all_channel / 4), kernel_size=3, dilation=3, padding=3)
        self.dconv3 = BasicConv2d(all_channel, int(all_channel / 4), kernel_size=3, dilation=5, padding=5)
        self.dconv4 = BasicConv2d(all_channel, int(all_channel / 4), kernel_size=3, dilation=7, padding=7)
        self.fuse_dconv = nn.Conv2d(all_channel, all_channel, kernel_size=3, padding=1)


    def forward(self, x, ir):
        multiplication = self.conv1(x * ir)
        summation = self.conv2(x + ir)
        fusion = (summation + multiplication)
        x1 = self.dconv1(fusion)
        x2 = self.dconv2(fusion)
        x3 = self.dconv3(fusion)
        x4 = self.dconv4(fusion)
        out = self.fuse_dconv(torch.cat((x1, x2, x3, x4), dim=1))

        return out

class GCM1(nn.Module):
    def __init__(self, in_channel, out_channel, residual=True, expand_ratio=4):
        super(GCM1, self).__init__()

        hidden_dim = int(round(out_channel * expand_ratio))

        self.use_res_connect = residual

        self.conv1 = ConvBNReLU(in_channel, hidden_dim, ksize=1, pad=0, prelu=False)

        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            ConvBNReLU(in_channel, hidden_dim, 1),
        )
        self.branch1 = nn.Sequential(
            ConvBNReLU(in_channel, hidden_dim, 1),
            ConvBNReLU(hidden_dim, hidden_dim, ksize=(1, 3), pad=(0, 1), groups=hidden_dim),
            ConvBNReLU(hidden_dim, hidden_dim, ksize=(3, 1), pad=(1, 0), groups=hidden_dim),
            ConvBNReLU(hidden_dim, hidden_dim, 3, pad=3, dilation=3, groups=hidden_dim)
        )
        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channel, hidden_dim, 1),
            ConvBNReLU(hidden_dim, hidden_dim, ksize=(1, 5), pad=(0, 2), groups=hidden_dim),
            ConvBNReLU(hidden_dim, hidden_dim, ksize=(5, 1), pad=(2, 0), groups=hidden_dim),
            ConvBNReLU(hidden_dim, hidden_dim, 3, pad=5, dilation=5, groups=hidden_dim)
        )
        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channel, hidden_dim, 1),
            ConvBNReLU(hidden_dim, hidden_dim, ksize=(1, 7), pad=(0, 3), groups=hidden_dim),
            ConvBNReLU(hidden_dim, hidden_dim, ksize=(7, 1), pad=(3, 0), groups=hidden_dim),
            ConvBNReLU(hidden_dim, hidden_dim, 3, pad=7, dilation=7, groups=hidden_dim)
        )
        self.conv_cat = ConvBNReLU(4 * hidden_dim, hidden_dim, 3, pad=0)
        self.conv_res = ConvBNReLU(in_channel, out_channel, 3, use_relu=False)

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.out_conv(self.conv_cat(torch.cat((x0, x1, x2,x3), 1)))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class GCM2(nn.Module):
    def __init__(self, in_channel, out_channel, residual=True, expand_ratio=4):
        super(GCM2, self).__init__()

        hidden_dim = int(round(out_channel * expand_ratio))

        self.use_res_connect = residual

        self.conv1 = ConvBNReLU(in_channel, hidden_dim, ksize=1, pad=0, prelu=False)

        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            ConvBNReLU(in_channel, hidden_dim, 1),
        )
        self.branch1 = nn.Sequential(
            ConvBNReLU(in_channel, hidden_dim, 1),
            ConvBNReLU(hidden_dim, hidden_dim, ksize=(1, 3), pad=(0, 1), groups=hidden_dim),
            ConvBNReLU(hidden_dim, hidden_dim, ksize=(3, 1), pad=(1, 0), groups=hidden_dim),
            ConvBNReLU(hidden_dim, hidden_dim, 3, pad=3, dilation=3, groups=hidden_dim)
        )
        # self.branch2 = nn.Sequential(
        #     ConvBNReLU(in_channel, hidden_dim, 1),
        #     ConvBNReLU(hidden_dim, hidden_dim, ksize=(1, 5), pad=(0, 2), groups=hidden_dim),
        #     ConvBNReLU(hidden_dim, hidden_dim, ksize=(5, 1), pad=(2, 0), groups=hidden_dim),
        #     ConvBNReLU(hidden_dim, hidden_dim, 3, pad=5, dilation=5, groups=hidden_dim)
        # )
        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channel, hidden_dim, 1),
            ConvBNReLU(hidden_dim, hidden_dim, ksize=(1, 7), pad=(0, 3), groups=hidden_dim),
            ConvBNReLU(hidden_dim, hidden_dim, ksize=(7, 1), pad=(3, 0), groups=hidden_dim),
            ConvBNReLU(hidden_dim, hidden_dim, 3, pad=7, dilation=7, groups=hidden_dim)
        )
        self.conv_cat = ConvBNReLU(3 * hidden_dim, hidden_dim, 3, pad=0)
        self.conv_res = ConvBNReLU(in_channel, out_channel, 3, use_relu=False)

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        # x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.out_conv(self.conv_cat(torch.cat((x0, x1, x3), 1)))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class GCM3(nn.Module):
    def __init__(self, in_channel, out_channel, residual=True, expand_ratio=4):
        super(GCM3, self).__init__()

        hidden_dim = int(round(out_channel * expand_ratio))

        self.use_res_connect = residual

        self.conv1 = ConvBNReLU(in_channel, hidden_dim, ksize=1, pad=0, prelu=False)

        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            ConvBNReLU(in_channel, hidden_dim, 1),
        )
        self.branch1 = nn.Sequential(
            ConvBNReLU(in_channel, hidden_dim, 1),
            ConvBNReLU(hidden_dim, hidden_dim, ksize=(1, 3), pad=(0, 1), groups=hidden_dim),
            ConvBNReLU(hidden_dim, hidden_dim, ksize=(3, 1), pad=(1, 0), groups=hidden_dim),
            ConvBNReLU(hidden_dim, hidden_dim, 3, pad=3, dilation=3, groups=hidden_dim)
        )
        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channel, hidden_dim, 1),
            ConvBNReLU(hidden_dim, hidden_dim, ksize=(1, 5), pad=(0, 2), groups=hidden_dim),
            ConvBNReLU(hidden_dim, hidden_dim, ksize=(5, 1), pad=(2, 0), groups=hidden_dim),
            ConvBNReLU(hidden_dim, hidden_dim, 3, pad=5, dilation=5, groups=hidden_dim)
        )
        # self.branch3 = nn.Sequential(
        #     ConvBNReLU(in_channel, hidden_dim, 1),
        #     ConvBNReLU(hidden_dim, hidden_dim, ksize=(1, 7), pad=(0, 3), groups=hidden_dim),
        #     ConvBNReLU(hidden_dim, hidden_dim, ksize=(7, 1), pad=(3, 0), groups=hidden_dim),
        #     ConvBNReLU(hidden_dim, hidden_dim, 3, pad=7, dilation=7, groups=hidden_dim)
        # )
        self.conv_cat = ConvBNReLU(3 * hidden_dim, hidden_dim, 3, pad=0)
        self.conv_res = ConvBNReLU(in_channel, out_channel, 3, use_relu=False)

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        # x3 = self.branch3(x)

        x_cat = self.out_conv(self.conv_cat(torch.cat((x0, x1,x2), 1)))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class MAGNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.rgb_backbone = smt_t(pretrained)
        self.d_backbone = mobilenet_v2(pretrained)
        # Fuse
       # self.gfm2 = CRF(512)
        self.SCA3 = SCA(256)
        self.SCA2 = SCA(128)
        self.EDS1 = EDS(64)
        self.gfm2 = GFM(inc=512, expend_ratio=2)
     #   self.gfm1 = GFM(inc=256, expend_ratio=3)
     #   self.mafm2 = MAFM(inc=128)
     #   self.mafm1 = MAFM(inc=64)
        # Pred
        self.mcm1 = HighLevelVisualDecoder(inc=512, outc=256)
        self.mcm2 = LowLevelVisualDecoder(inc=256, outc=128)
        self.mcm3 = LowLevelVisualDecoder(inc=128, outc=64)

        self.d_trans_4 = Trans(320, 512)
        self.d_trans_3 = Trans(96, 256)
        self.d_trans_2 = Trans(32, 128)
        self.d_trans_1 = Trans(24, 64)
        enc_channels = [128, 256,512]
        self.GCM1 = GCM1(enc_channels[-3], 32)
        self.GCM2 = GCM2(enc_channels[-2], 32)
        self.GCM3 = GCM3(enc_channels[-1], 32)
        self.agint = aggregation_init(32)

        self.predtrans = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, groups=512),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1)
        )
        self.ram_saliency = RAMOptimized(
            input_channels=1,  # 与 fea_* 的通道数一致
            hidden_channels=[64],  # 隐藏通道数
            kernel_size=3,  # 卷积核大小
        )
        for m in self.modules():
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('BatchNorm') != -1:
                if hasattr(m, 'weight'):
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.bias, 0)



    def forward(self, x_rgb, x_d):
        # rgb
        _, (rgb_1, rgb_2, rgb_3, rgb_4)= self.rgb_backbone(x_rgb)
        # d
        _, d_1, d_2, d_3, d_4 = self.d_backbone(x_d)

        d_4 = self.d_trans_4(d_4)
        d_3 = self.d_trans_3(d_3)
        d_2 = self.d_trans_2(d_2)
        d_1 = self.d_trans_1(d_1)

        # Fuse
       # fuse_4 = self.RFN4(rgb_4, d_4)  # [B, 512, 12, 12]
        fuse_3 = self.SCA3(rgb_3, d_3)  # [B, 256, 24, 24]
        fuse_2 = self.SCA2(rgb_2, d_2)  # [B, 128, 48, 48]
        fuse_1 = self.EDS1(rgb_1, d_1)  # [B, 64, 96, 96]

        fuse_4 = self.gfm2(rgb_4, d_4)  # [B, 512, 12, 12]
       # fuse_3 = self.gfm1(rgb_3, d_3)  # [B, 256, 24, 24]
       # fuse_2 = self.mafm2(rgb_2, d_2)  # [B, 128, 48, 48]
       # fuse_1 = self.mafm1(rgb_1, d_1)  # [B, 64, 96, 96]

        pred_4 = F.interpolate(self.predtrans(fuse_4), TRAIN_SIZE, mode="bilinear", align_corners=True)
        pred_3, xf_3 = self.mcm1(fuse_3, fuse_4)
        pred_2, xf_2 = self.mcm2(fuse_2, xf_3)
        pred_1, xf_1 = self.mcm3(fuse_1, xf_2)

        return pred_1, pred_2, pred_3, pred_4

class Trans(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.trans = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )
        self.apply(self._init_weights)

    def forward(self, d):
        return self.trans(d)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

class COI(nn.Module):
    def __init__(self, inc, k=3, p=1):
        super().__init__()
        self.outc = inc
        self.dw = nn.Conv2d(inc, self.outc, kernel_size=k, padding=p, groups=inc)
        self.conv1_1 = nn.Conv2d(inc, self.outc, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(self.outc)
        self.bn2 = nn.BatchNorm2d(self.outc)
        self.bn3 = nn.BatchNorm2d(self.outc)
        self.act = nn.GELU()
        self.apply(self._init_weights)

    def forward(self, x):
        shortcut = self.bn1(x)

        x_dw = self.bn2(self.dw(x))
        x_conv1_1 = self.bn3(self.conv1_1(x))
        return self.act(shortcut + x_dw + x_conv1_1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class MHMC(nn.Module):
    def __init__(self, dim, ca_num_heads=4, qkv_bias=True, proj_drop=0., ca_attention=1, expand_ratio=2):
        super().__init__()

        self.ca_attention = ca_attention
        self.dim = dim
        self.ca_num_heads = ca_num_heads

        assert dim % ca_num_heads == 0, f"dim {dim} should be divided by num_heads {ca_num_heads}."

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.split_groups = self.dim // ca_num_heads

        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.s = nn.Linear(dim, dim, bias=qkv_bias)
        for i in range(self.ca_num_heads):
            local_conv = nn.Conv2d(dim // self.ca_num_heads, dim // self.ca_num_heads, kernel_size=(3 + i * 2),
                                   padding=(1 + i), stride=1,
                                   groups=dim // self.ca_num_heads)  # kernel_size 3,5,7,9 大核dw卷积，padding 1,2,3,4
            setattr(self, f"local_conv_{i + 1}", local_conv)
        self.proj0 = nn.Conv2d(dim, dim * expand_ratio, kernel_size=1, padding=0, stride=1,
                               groups=self.split_groups)
        self.bn = nn.BatchNorm2d(dim * expand_ratio)
        self.proj1 = nn.Conv2d(dim * expand_ratio, dim, kernel_size=1, padding=0, stride=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        v = self.v(x)
        s = self.s(x).reshape(B, H, W, self.ca_num_heads, C // self.ca_num_heads).permute(3, 0, 4, 1,
                                                                                          2)  # num_heads,B,C,H,W
        for i in range(self.ca_num_heads):
            local_conv = getattr(self, f"local_conv_{i + 1}")
            s_i = s[i]  # B,C,H,W
            s_i = local_conv(s_i).reshape(B, self.split_groups, -1, H, W)
            if i == 0:
                s_out = s_i
            else:
                s_out = torch.cat([s_out, s_i], 2)
        s_out = s_out.reshape(B, C, H, W)
        s_out = self.proj1(self.act(self.bn(self.proj0(s_out))))
        self.modulator = s_out
        s_out = s_out.reshape(B, C, N).permute(0, 2, 1)
        x = s_out * v

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SAttention(nn.Module):
    def __init__(self, dim, sa_num_heads=8, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.sa_num_heads = sa_num_heads

        assert dim % sa_num_heads == 0, f"dim {dim} should be divided by num_heads {sa_num_heads}."

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        head_dim = dim // sa_num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.sa_num_heads, C // self.sa_num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.sa_num_heads, C // self.sa_num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) + \
            self.local_conv(v.transpose(1, 2).reshape(B, N, C).transpose(1, 2).view(B, C, H, W)).view(B, C,
                                                                                                      N).transpose(1, 2)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x.permute(0, 2, 1).reshape(B, C, H, W)

def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )

class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout

class FrozenBatchNorm2d(nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        # Cast all fixed parameters to half() if necessary
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "{})".format(self.weight.shape[0])
        return s

from torch.nn import BatchNorm2d

class ConvBNReLU(nn.Module):
    def __init__(self, nIn, nOut, ksize=3, stride=1, pad=1, dilation=1, groups=1,
                 bias=True, use_relu=True, leaky_relu=False, use_bn=True, frozen=False,
                 prelu=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=ksize, stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=bias)
        if use_bn:
            if frozen:
                self.bn = FrozenBatchNorm2d(nOut)
            else:
                self.bn = BatchNorm2d(nOut)
        else:
            self.bn = None
        if use_relu:
            if leaky_relu is True:
                self.act = nn.LeakyReLU(0.1, inplace=True)
            elif prelu is True:
                self.act = nn.PReLU(nOut)
            else:
                self.act = nn.ReLU(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        # 打印输入的尺寸

        # 卷积操作
        x = self.conv(x)

        # 批归一化操作
        if self.bn is not None:
            x = self.bn(x)

        # 激活函数操作
        if self.act is not None:
            x = self.act(x)

        return x

class aggregation_init(nn.Module):

    def __init__(self, channel):
        super(aggregation_init, self).__init__()
        self.relu = nn.ReLU(True)

        hidden_dim=channel*4
        self.convx1 = ConvBNReLU(channel, hidden_dim, ksize=1, pad=0, prelu=False)
        self.convx2 = ConvBNReLU(channel, hidden_dim, ksize=1, pad=0, prelu=False)
        self.convx3 = ConvBNReLU(channel, hidden_dim, ksize=1, pad=0, prelu=False)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = ConvBNReLU(hidden_dim, hidden_dim, 3, pad=1,groups=hidden_dim)
        self.conv_upsample2 = ConvBNReLU(hidden_dim, hidden_dim, 3, pad=1,groups=hidden_dim)
        self.conv_upsample3 = ConvBNReLU(hidden_dim, hidden_dim, 3, pad=1,groups=hidden_dim)
        self.conv_upsample4 = ConvBNReLU(2 * hidden_dim, 2 * hidden_dim, 3, pad=1,groups=hidden_dim)
        self.conv_upsample5 = ConvBNReLU(hidden_dim, hidden_dim, 3, pad=1,groups=hidden_dim)
        self.conv_upsample6 = ConvBNReLU(hidden_dim, hidden_dim, 3, pad=1,groups=hidden_dim)
        self.conv_upsample7 = ConvBNReLU(hidden_dim, hidden_dim, 3, pad=1,groups=hidden_dim)
        self.conv_concat2 = ConvBNReLU(2 * hidden_dim, 2 * hidden_dim, 3, pad=1,groups=hidden_dim)
        self.conv_concat3 = ConvBNReLU(3 * hidden_dim, 3 * hidden_dim, 3, pad=1,groups=hidden_dim)

        self.conv4 = ConvBNReLU(3 * hidden_dim, 3 * hidden_dim, 3, pad=1,groups=hidden_dim)
        # self.conv4_1 = ConvBNReLU(3 * hidden_dim, 2 * hidden_dim, 3, pad=1,groups=hidden_dim)
        # self.conv4_2 = ConvBNReLU(2 * hidden_dim, hidden_dim, 3, pad=1,groups=hidden_dim)

        #self.conv4_0=ConvBNReLU(3*hidden_dim,channel,1)
        self.conv4_0 = ConvBNReLU(3 * hidden_dim, channel, ksize=1, stride=1, pad=0)
        self.conv5=ConvBNReLU(channel,channel//2,1)
        self.conv5_0 = nn.Conv2d(channel//2, 1, 1)

        # self.x1_2_3_cat=BasicConv2d(channel*2,channel,3,padding=1)

    def forward(self, x1, x2, x3):
        x1=self.convx1(x1)
        x2=self.convx1(x2)
        x3=self.convx1(x3)


        x1_1 = self.conv_upsample1(x1)
        x1_1 = self.upsample(x1_1)
        x1_2 = self.conv_upsample2(self.upsample(x1))

        x2_1 = x2 * x1_2
        x2_2 = self.conv_upsample3(x2_1)

        x21 = torch.cat((x2_2, x1_1), dim=1)
        x21 = self.conv_concat2(x21)
        x21 = self.conv_upsample4(x21)
        x21 = self.upsample(x21)

        x1_3 = self.upsample(self.upsample(self.conv_upsample5(x1)))
        x2_3 = self.upsample(self.conv_upsample6(x2))

        # x3_1=x3*self.x1_2_3_cat(torch.cat((x1_3,x2_3),dim=1))
        x3_1 = x3 * x1_3 * x2_3
        x3_2 = self.conv_upsample7(x3_1)

        x321 = torch.cat((x21, x3_2), dim=1)
        x = self.conv_concat3(x321)

        x = self.conv4(x)
        print(x.shape)
        x_att_channel_ori = self.conv4_0(x)
        print(x_att_channel_ori.shape)


        return x_att_channel_ori
# Decoder
class LowLevelVisualDecoder(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.rc = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=3, padding=1, stride=1, groups=inc),
            nn.BatchNorm2d(inc),
            nn.ReLU(),
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1),
            nn.BatchNorm2d(outc),
            nn.ReLU()
        )
        self.predtrans = nn.Sequential(
            nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=3, padding=1, groups=outc),
            nn.BatchNorm2d(outc),
            nn.ReLU(),
            nn.Conv2d(in_channels=outc, out_channels=1, kernel_size=1)
        )

        self.rc2 = nn.Sequential(
            nn.Conv2d(in_channels=outc * 2, out_channels=outc * 2, kernel_size=3, padding=1, groups=outc * 2),
            nn.BatchNorm2d(outc * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=outc * 2, out_channels=outc, kernel_size=1, stride=1),
            nn.BatchNorm2d(outc),
            nn.ReLU()
        )

        self.apply(self._init_weights)

    def forward(self, x1, x2):
        x2_upsample = self.upsample2(x2)  # 上采样
        x2_rc = self.rc(x2_upsample)  # 减少通道数
        shortcut = x2_rc

        x_cat = torch.cat((x1, x2_rc), dim=1)  # 拼接
        x_forward = self.rc2(x_cat)  # 减少通道数2
        x_forward = x_forward + shortcut
        pred = F.interpolate(self.predtrans(x_forward), TRAIN_SIZE, mode="bilinear", align_corners=True)  # 预测图

        return pred, x_forward

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

class HighLevelVisualDecoder(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()

        # 上采样模块 (Upsampling)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # 高级视觉特征解码部分1 (First Stage of High-level Visual Decoder)
        self.rc = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=inc * 2, kernel_size=3, padding=1, stride=1, groups=inc),  # 增加通道数
            nn.BatchNorm2d(inc * 2),
            nn.GELU(),
            nn.Conv2d(in_channels=inc * 2, out_channels=outc, kernel_size=1, stride=1),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )

        # 高级特征变换 (Transform High-level Features)
        self.predtrans = nn.Sequential(
            nn.Conv2d(in_channels=outc, out_channels=outc * 2, kernel_size=3, padding=1, groups=outc),  # 增加通道
            nn.BatchNorm2d(outc * 2),
            nn.GELU(),
            nn.Conv2d(in_channels=outc * 2, out_channels=1, kernel_size=1)  # 输出单通道图像
        )

        # 高级特征解码部分2 (Second Stage of High-level Visual Decoder)
        self.rc2 = nn.Sequential(
            nn.Conv2d(in_channels=outc * 2, out_channels=outc * 2, kernel_size=3, padding=1, groups=outc * 2),  # 增加通道
            nn.BatchNorm2d(outc * 2),
            nn.GELU(),
            nn.Conv2d(in_channels=outc * 2, out_channels=outc, kernel_size=1, stride=1),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )
        self.apply(self._init_weights)
    def forward(self, x1, x2):
        # 上采样并处理
        x2_upsample = self.upsample2(x2)  # 上采样 x2
        x2_rc = self.rc(x2_upsample)  # 处理 x2，减少通道数
        shortcut = x2_rc  # 存储残差
        x_cat = torch.cat((x1, x2_rc), dim=1)  # 拼接
        x_forward = self.rc2(x_cat)  # 高级特征解码部分2
        x_forward = x_forward + shortcut  # 加入残差连接

        # 预测并上采样
        pred = F.interpolate(self.predtrans(x_forward), size=TRAIN_SIZE, mode="bilinear", align_corners=True)

        return pred, x_forward

    def _init_weights(self, m):
        # 权重初始化
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

# Global Fusion Module
class GFM(nn.Module):
    def __init__(self, inc, expend_ratio=2):
        super().__init__()
        self.expend_ratio = expend_ratio
        assert expend_ratio in [2, 3], f"expend_ratio {expend_ratio} mismatch"

        self.sa = SAttention(dim=inc)
        self.dw_pw = DWPWConv(inc * expend_ratio, inc)
        self.act = nn.GELU()
        self.apply(self._init_weights)

    def forward(self, x, d):
        B, C, H, W = x.shape
        if self.expend_ratio == 2:
            cat = torch.cat((x, d), dim=1)
        else:
            multi = x * d
            cat = torch.cat((x, d, multi), dim=1)
        x_rc = self.dw_pw(cat).flatten(2).permute(0, 2, 1)
        x_ = self.sa(x_rc, H, W)
        x_ = x_ + x
        return self.act(x_)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

class DWPWConv(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=3, padding=1, stride=1, groups=inc),
            nn.BatchNorm2d(inc),
            nn.GELU(),
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)

import time

if __name__ == '__main__':
    # 初始化模型
    model = MAGNet().to('cuda')
    # 创建输入张量
    r = torch.randn([1, 3, 384, 384]).to('cuda')
    d = torch.randn([1, 3, 384, 384]).to('cuda')

    # 计算模型的参数量和FLOPs
    params, flops = profile(model, inputs=(r, d))
    flops, params = clever_format([flops, params], "%.2f")
    print(f"FLOPs: {flops}, Params: {params}")

    # 测试FPS
    num_runs = 100  # 测试100次推理
    start_time = time.time()  # 记录开始时间

    for _ in range(num_runs):
        with torch.no_grad():  # 在测试时不计算梯度
            model(r, d)

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算总耗时
    fps = num_runs / elapsed_time  # 计算FPS

    print(f"FPS: {fps:.2f}")
