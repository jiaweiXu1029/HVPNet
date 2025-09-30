import math
from model.smt import smt_t,smt_b,smt_l
from model.MobileNetV2 import mobilenet_v2
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import numpy as np
from model.HolisticAttention import HA
from SRA import SRA  # 假设已经正确导入SRA模块
from thop import profile, clever_format
import time
import torch
from torch.nn import BatchNorm2d
import torch.nn as nn
from timm.layers import LayerNorm2d

TRAIN_SIZE = 384

class RFB(nn.Module):
    # RFB-like multi-scale module
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

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

class RSU(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin

def _upsample_like(src, tar):
    src = F.upsample(src, size=tar.shape[2:], mode='bilinear')

    return src

class MCM(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.rc = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=3, padding=1, stride=1, groups=inc),
            nn.BatchNorm2d(inc),
            nn.GELU(),
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )
        self.predtrans = nn.Sequential(
            nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=3, padding=1, groups=outc),
            nn.BatchNorm2d(outc),
            nn.GELU(),
            nn.Conv2d(in_channels=outc, out_channels=1, kernel_size=1)
        )

        self.rc2 = nn.Sequential(
            nn.Conv2d(in_channels=outc * 2, out_channels=outc * 2, kernel_size=3, padding=1, groups=outc * 2),
            nn.BatchNorm2d(outc * 2),
            nn.GELU(),
            nn.Conv2d(in_channels=outc * 2, out_channels=outc, kernel_size=1, stride=1),
            nn.BatchNorm2d(outc),
            nn.GELU()
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

class Net(nn.Module):
    def __init__(self, pretrained=True,channel=32):
        super().__init__()
        self.rgb_backbone = smt_t(pretrained)
        self.d_backbone = mobilenet_v2(pretrained)

        # Fuse
        self.gfm2 = GFM(inc=512, expend_ratio=2)
        self.SCA3 = SCA(256)
        self.SCA2 = SCA(128)
        self.EDS = EDS(64)
        # Pred
        self.mcm1 = MCM(inc=128, outc=64)
        self.rfb4_1 = RFB(512, channel)
        self.rfb3_1 = RFB(256, channel)
        self.rfb2_1 = RFB(128, channel)
        self.agg1 = aggregation(channel)
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.HA = HA()

        self.d_trans_4 = Trans(320, 512)
        self.d_trans_3 = Trans(96, 256)
        self.d_trans_2 = Trans(32, 128)
        self.d_trans_1 = Trans(24, 64)


        self.predtrans4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, groups=512),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1)
        )
        self.predtrans3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, groups=256),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1)
        )
        self.predtrans2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1)
        )
        self.predtrans1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1)
        )

    def forward(self, x_rgb, x_d):
        # rgb
        _, (rgb_1, rgb_2, rgb_3, rgb_4) = self.rgb_backbone(x_rgb)
        # d
        _, d_1, d_2, d_3, d_4 = self.d_backbone(x_d)
        d_4 = self.d_trans_4(d_4)
        d_3 = self.d_trans_3(d_3)
        d_2 = self.d_trans_2(d_2)
        d_1 = self.d_trans_1(d_1)

        # Fuse
        fuse_4 = self.gfm2(rgb_4, d_4)  # [B, 512, 12, 12]
        pred_4 = F.interpolate(self.predtrans4(fuse_4), TRAIN_SIZE, mode="bilinear", align_corners=True)
        fuse_4=self.rfb4_1(fuse_4)
        fuse_3 = self.SCA3(rgb_3, d_3)  # [B, 256, 24, 24]
        pred_3 = F.interpolate(self.predtrans3(fuse_3), TRAIN_SIZE, mode="bilinear", align_corners=True)
        fuse_3 = self.rfb3_1(fuse_3)
        fuse_2 = self.SCA2(rgb_2, d_2)  # [B, 128, 48, 48]
        x2=fuse_2
        fuse_2 = self.rfb2_1(fuse_2)
        attention_map = self.agg1(fuse_4, fuse_3, fuse_2)
        fuse_2 = self.HA(attention_map.sigmoid(), x2)
        pred_2 = F.interpolate(self.predtrans2(fuse_2), TRAIN_SIZE, mode="bilinear", align_corners=True)
        fuse_1 = self.EDS(rgb_1, d_1)  # [B, 64, 96, 96]

        # Pred
        pred_1, xf_1 = self.mcm1(fuse_1, fuse_2)

        return pred_1,pred_2, pred_3, pred_4

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

# Conv_One_Identity
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

# Multi-scale Awareness Fusion Module

class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation model, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x

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

if __name__ == '__main__':
    # 初始化模型
    model = Net().to('cuda')
    # 创建输入张量
    r = torch.randn([1, 3, 384, 384]).to('cuda')
    d = torch.randn([1, 3, 384, 384]).to('cuda')
    # 计算模型的参数量和FLOPs
    params, flops = profile(model, inputs=(r, d))
    flops, params = clever_format([flops, params], "%.2f")
    print(f"FLOPs: {flops}, Params: {params}")
    num_runs = 1  # 测试100次推理
    start_time = time.time()  # 记录开始时间

    for _ in range(num_runs):
        with torch.no_grad():  # 在测试时不计算梯度
            model(r, d)

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算总耗时
    fps = num_runs / elapsed_time  # 计算FPS

    print(f"FPS: {fps:.2f}")
