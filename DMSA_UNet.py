import torch
import torch.nn as nn
# from einops import rearrange
# from einops.layers.torch import Rearrange
from torch.nn import functional as F
from typing import Tuple
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        tx = x.transpose(1, 2).view(B, C, H, W)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(2).transpose(1, 2)


class MixFFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.dwconv(self.fc1(x), H, W))
        out = self.fc2(ax)
        return out


class Context_GLU(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        x = self.fc1(x)
        ax = self.act(self.norm1(self.dwconv(x, H, W) * x))
        out = self.fc2(ax)
        return out


class MLP_FFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Aggregator(nn.Module):
    def __init__(self, dim, seg=4):
        super().__init__()
        self.dim = dim
        self.seg = seg

        seg_dim = self.dim // self.seg

        self.norm0 = nn.BatchNorm2d(seg_dim)
        self.act0 = nn.ReLU()

        self.agg1 = SeparableConv2d(seg_dim, seg_dim, 3, 1, 1)
        self.norm1 = nn.BatchNorm2d(seg_dim)
        self.act1 = nn.ReLU()

        self.agg2 = SeparableConv2d(seg_dim, seg_dim, 5, 1, 2)
        self.norm2 = nn.BatchNorm2d(seg_dim)
        self.act2 = nn.ReLU()

        self.agg3 = SeparableConv2d(seg_dim, seg_dim, 7, 1, 3)
        self.norm3 = nn.BatchNorm2d(seg_dim)
        self.act3 = nn.ReLU()

    def forward(self, x, size, num_head):
        B, N, C = x.shape
        H, W = size
        assert N == H * W

        x = x.transpose(1, 2).view(B, C, H, W)
        seg_dim = self.dim // self.seg

        x = x.split([seg_dim]*self.seg, dim=1)

        x0 = self.act0(self.norm0(x[0].contiguous()))
        x1 = self.act1(self.norm1(self.agg1(x[1].contiguous())))
        x2 = self.act2(self.norm2(self.agg2(x[2].contiguous())))
        x3 = self.act3(self.norm3(self.agg3(x[3].contiguous())))

        x0 = x0.reshape(2, B // 2, num_head, seg_dim // num_head, H * W)
        x1 = x1.reshape(2, B // 2, num_head, seg_dim // num_head, H * W)
        x2 = x2.reshape(2, B // 2, num_head, seg_dim // num_head, H * W)
        x3 = x3.reshape(2, B // 2, num_head, seg_dim // num_head, H * W)

        x = torch.cat([x0, x1, x2, x3], dim=3).permute(0, 1, 2, 4, 3)

        return x


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.pointwise_conv(self.conv1(x))
        return x


class SpatialAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,):
        super().__init__()

        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)  # Note: attn_drop is actually not used.
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.aggregator = Aggregator(dim=dim, seg=4)

    def forward(self, x, size):
        B, N, C = x.shape

        # Q, K, V.
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q = qkv[0].reshape(B, N, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3)
        kv = qkv[1:].reshape(2*B, N, C)
        kv = self.aggregator(kv, size, self.num_heads)
        k, v = kv[0], kv[1]

        # att
        k_softmax = k.softmax(dim=-2)
        k_softmax_T_dot_v = torch.einsum('b h n k, b h n v -> b h k v', k_softmax, v)
        q_softmax = q.softmax(dim=-1)
        eff_att = torch.einsum('b h n k, b h k v -> b h n v', q_softmax, k_softmax_T_dot_v)
        # Merge and reshape.
        x = eff_att
        x = x.transpose(1, 2).reshape(B, N, C)

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class ChannelAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,):
        super().__init__()

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)  # Note: attn_drop is actually not used.
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.aggregator = Aggregator(dim=dim, seg=4)

    def forward(self, x, size):
        B, N, C = x.shape

        # Q, K, V.
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q = qkv[0].reshape(B, N, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3)
        kv = qkv[1:].reshape(2*B, N, C)
        kv = self.aggregator(kv, size, self.num_heads)
        k, v = kv[0], kv[1]

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # -------------------
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        # ------------------
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


# class FullChannelAttention(nn.Module):
#     def __init__(self, reduction_ratio, dims, dim, num_heads=1, qkv_bias=False, attn_drop=0., proj_drop=0.,):
#         super().__init__()
#
#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
#
#         self.q = nn.Linear(dim, dim, bias=qkv_bias)
#         self.k = nn.Linear(dim, dim, bias=qkv_bias)
#         self.v = nn.Linear(dim, dim, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)  # Note: attn_drop is actually not used.
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#         # self.aggregator = Aggregator(dim=dim, seg=4)
#
#         self.reduction_ratio = reduction_ratio
#         if len(self.reduction_ratio) == 3:
#             self.k0 = nn.Conv2d(dims[0], dims[0], reduction_ratio[0], reduction_ratio[0])
#             self.k1 = nn.Conv2d(dims[1], dims[1], reduction_ratio[1], reduction_ratio[1])
#             self.k2 = nn.Conv2d(dims[2], dims[2], reduction_ratio[2], reduction_ratio[2])
#             self.v0 = nn.Conv2d(dims[0], dims[0], reduction_ratio[0], reduction_ratio[0])
#             self.v1 = nn.Conv2d(dims[1], dims[1], reduction_ratio[1], reduction_ratio[1])
#             self.v2 = nn.Conv2d(dims[2], dims[2], reduction_ratio[2], reduction_ratio[2])
#         elif len(self.reduction_ratio) == 2:
#             self.k0 = nn.Conv2d(dims[0], dims[0], reduction_ratio[0], reduction_ratio[0])
#             self.k1 = nn.Conv2d(dims[1], dims[1], reduction_ratio[1], reduction_ratio[1])
#             self.v0 = nn.Conv2d(dims[0], dims[0], reduction_ratio[0], reduction_ratio[0])
#             self.v1 = nn.Conv2d(dims[1], dims[1], reduction_ratio[1], reduction_ratio[1])
#         elif len(self.reduction_ratio) == 1:
#             self.k0 = nn.Conv2d(dims[0], dims[0], reduction_ratio[0], reduction_ratio[0])
#             self.v0 = nn.Conv2d(dims[0], dims[0], reduction_ratio[0], reduction_ratio[0])
#
#     def forward(self, query, key_value):
#         B, N, C = query.shape
#
#         # Q,
#         q = self.q(query).reshape(B, N, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3)
#         # K, V.
#         k = []
#         v = []
#         if len(self.reduction_ratio) == 3:
#             k.append(self.k0(key_value[0]).reshape(B, self.num_heads, -1, N).permute(0, 1, 3, 2))
#             k.append(self.k1(key_value[1]).reshape(B, self.num_heads, -1, N).permute(0, 1, 3, 2))
#             k.append(self.k2(key_value[2]).reshape(B, self.num_heads, -1, N).permute(0, 1, 3, 2))
#             v.append(self.v0(key_value[0]).reshape(B, self.num_heads, -1, N).permute(0, 1, 3, 2))
#             v.append(self.v1(key_value[1]).reshape(B, self.num_heads, -1, N).permute(0, 1, 3, 2))
#             v.append(self.v2(key_value[2]).reshape(B, self.num_heads, -1, N).permute(0, 1, 3, 2))
#         elif len(self.reduction_ratio) == 2:
#             k.append(self.k0(key_value[0]).reshape(B, self.num_heads, -1, N).permute(0, 1, 3, 2))
#             k.append(self.k1(key_value[1]).reshape(B, self.num_heads, -1, N).permute(0, 1, 3, 2))
#             v.append(self.v0(key_value[0]).reshape(B, self.num_heads, -1, N).permute(0, 1, 3, 2))
#             v.append(self.v1(key_value[1]).reshape(B, self.num_heads, -1, N).permute(0, 1, 3, 2))
#         elif len(self.reduction_ratio) == 1:
#             k.append(self.k0(key_value[0]).reshape(B, self.num_heads, -1, N).permute(0, 1, 3, 2))
#             v.append(self.v0(key_value[0]).reshape(B, self.num_heads, -1, N).permute(0, 1, 3, 2))
#
#         if len(self.reduction_ratio) > 0:
#             k.append(self.k(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3))
#             v.append(self.v(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3))
#             k = torch.cat(k, dim=-1)
#             v = torch.cat(v, dim=-1)
#         else:
#             k = self.k(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
#             v = self.v(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
#
#         q = q.transpose(-2, -1)
#         k = k.transpose(-2, -1)
#         v = v.transpose(-2, -1)
#
#         q = F.normalize(q, dim=-1)
#         k = F.normalize(k, dim=-1)
#
#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#         # -------------------
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
#         # ------------------
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         return x


class DualTransformerBlock(nn.Module):
    """
    Input  -> x (Size: (b, (H*W), d)), H, W
    Output -> (b, (H*W), d)
    """

    def __init__(self, in_dim, key_dim, value_dim, head_count=8, token_mlp="cglu"):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = SpatialAttention(in_dim, num_heads=head_count, attn_drop=0.1, proj_drop=0.1,)
        self.norm2 = nn.LayerNorm(in_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.channel_attn = ChannelAttention(in_dim, num_heads=head_count, attn_drop=0.1, proj_drop=0.1,)
        self.norm4 = nn.LayerNorm(in_dim)
        if token_mlp == "mix":
            self.mlp1 = MixFFN(in_dim, int(in_dim * 2))
            self.mlp2 = MixFFN(in_dim, int(in_dim * 2))
        elif token_mlp == "cglu":
            self.mlp1 = Context_GLU(in_dim, int(in_dim * 2))
            self.mlp2 = Context_GLU(in_dim, int(in_dim * 2))
        else:
            self.mlp1 = MLP_FFN(in_dim, int(in_dim * 2))
            self.mlp2 = MLP_FFN(in_dim, int(in_dim * 2))

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        # dual attention structure, efficient attention first then transpose attention
        norm1 = self.norm1(x)
        # norm1 = Rearrange("b (h w) d -> b d h w", h=H, w=W)(norm1)
        attn = self.attn(norm1, (H, W))
        # attn = Rearrange("b d h w -> b (h w) d")(attn)

        add1 = x + attn
        norm2 = self.norm2(add1)
        mlp1 = self.mlp1(norm2, H, W)

        add2 = add1 + mlp1
        norm3 = self.norm3(add2)
        channel_attn = self.channel_attn(norm3, (H, W))

        add3 = add2 + channel_attn
        norm4 = self.norm4(add3)
        mlp2 = self.mlp2(norm4, H, W)

        mx = add3 + mlp2
        return mx


class SPA(nn.Module):
    def __init__(self, kernel_size=7):
        super(SPA, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class EnDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EnDoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            SeparableConv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class DeDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DeDoubleConv, self).__init__()
        self.conv = nn.Sequential(
            SeparableConv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)


class DualTransformer(nn.Module):
    def __init__(self, in_dim, head_count=1, layers=1, token_mlp='cglu'):
        super(DualTransformer, self).__init__()
        self.block = nn.ModuleList(
            [DualTransformerBlock(in_dim, in_dim, in_dim, head_count, token_mlp) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).view(B, -1, C)
        for blk in self.block:
            x = blk(x, H, W)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


class DMSA_UNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DMSA_UNet, self).__init__()

        dims = [64, 128, 320, 512]

        self.conv1 = EnDoubleConv(in_ch, dims[0])
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = EnDoubleConv(dims[0], dims[1])
        self.pool2 = nn.MaxPool2d(2)

        self.tran3 = DualTransformer(dims[1], head_count=1, layers=1)
        self.conv3 = EnDoubleConv(dims[1], dims[2])
        self.pool3 = nn.MaxPool2d(2)

        self.tran4 = DualTransformer(dims[2], head_count=1, layers=1)
        self.conv4 = EnDoubleConv(dims[2], dims[3])
        self.pool4 = nn.MaxPool2d(2)

        self.tran5 = DualTransformer(dims[3], head_count=1, layers=1)

        self.up6 = nn.ConvTranspose2d(dims[3], dims[3], 2, stride=2)
        self.conv6 = DeDoubleConv(dims[3]*2, dims[3])
        self.tran6 = DualTransformer(dims[3], head_count=1, layers=1)

        self.up7 = nn.ConvTranspose2d(dims[3], dims[2], 2, stride=2)
        self.conv7 = DeDoubleConv(dims[2]*2, dims[2])
        self.tran7 = DualTransformer(dims[2], head_count=1, layers=1)

        self.up8 = nn.ConvTranspose2d(dims[2], dims[1], 2, stride=2)
        self.conv8 = DeDoubleConv(dims[1]*2, dims[1])

        self.up9 = nn.ConvTranspose2d(dims[1], dims[0], 2, stride=2)
        self.conv9 = DeDoubleConv(dims[0]*2, dims[0])

        self.conv10 = nn.Conv2d(dims[0], out_ch, 1)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        t3 = self.tran3(p2)
        c3 = self.conv3(t3)
        p3 = self.pool3(c3)

        t4 = self.tran4(p3)
        c4 = self.conv4(t4)
        p4 = self.pool4(c4)

        t5 = self.tran5(p4)

        up_6 = self.up6(t5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        t6 = self.tran6(c6)

        up_7 = self.up7(t6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        t7 = self.tran7(c7)

        up_8 = self.up8(t7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)

        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)

        c10 = self.conv10(c9)
        return c10


if __name__ == '__main__':
    inputs = torch.ones(1, 3, 224, 224)
    model = DMSA_UNet(3, 1)
    model.eval()
    from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis, flop_count_table

    flops = FlopCountAnalysis(model, inputs)
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    acts = ActivationCountAnalysis(model, inputs)

    print(f"total flops : {flops.total()}")
    print(f"total activations: {acts.total()}")
    print(f"number of parameter: {param}")

    print(flop_count_table(flops, max_depth=1))

    from thop import profile

    model = DMSA_UNet(3, 1)
    model.eval()
    flops, params = profile(model, inputs=(torch.ones(1, 3, 224, 224), ))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
