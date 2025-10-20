def parse_attention_modes(mode_str):
    """
    解析命令行传入的注意力组合字符串，如 'eca+lsa', 'dila+stage'
    返回：dict，标志哪些模块启用
    """
    if not mode_str:
        return {"ca": False, "cca": False, "eca": False, "sa": False, "esa": False, "lsa": False, "dila": False, "ula": False, "stage": False}
    mode_str = mode_str.lower()
    flags = {key: (key in mode_str) for key in ["ca", "cca", "eca", "sa", "esa", "lsa", "dila", "ula", "stage"]}
    return flags
"""
Unified Lightweight Refinement Network (ULRNet) for Efficient Image Super-Resolution:
- Gated Asymmetric Depthwise Conv (GADWConv)
- Unified Lightweight Attention (ULAttention)
- Learnable Residual Scaling (LRS)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import common


class ECA(nn.Module):
    def __init__(self, channel, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))
        return x * y.expand_as(x)


class ESA(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel // 4, 1)
        self.conv2 = nn.Conv2d(channel // 4, channel // 4, 3, padding=1)
        self.conv3 = nn.Conv2d(channel // 4, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv1(x)
        y = F.max_pool2d(y, kernel_size=7, stride=3)
        y = F.relu(self.conv2(y))
        y = F.interpolate(y, size=x.size()[2:], mode='bilinear', align_corners=False)
        y = self.sigmoid(self.conv3(y))
        return x * y


class LSA(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv = nn.Conv2d(channel, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att = self.sigmoid(self.conv(x))
        return x * att


class GADWConv(nn.Module):
    """
    GADWConv: Asymmetric depthwise conv with grouped 1x1 gating conv.
    The gate conv uses groups=4 when possible to reduce parameter cost.
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        # asymmetric depthwise: (1x3) then (3x1)
        self.dw1 = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1), groups=channels, bias=True)
        self.dw2 = nn.Conv2d(channels, channels, (3, 1), padding=(1, 0), groups=channels, bias=True)
        # use grouped 1x1 gating to reduce params when channels divisible by 4
        gate_groups = 4 if (channels % 4 == 0) else 1
        self.gate_conv = nn.Conv2d(channels, channels, 1, groups=gate_groups, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.dw1(x)
        out = self.dw2(out)
        gate = self.sigmoid(self.gate_conv(x))
        return out * gate + x


class CCAAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mid = max(8, channel // reduction)
        self.conv1 = nn.Conv2d(channel, mid, kernel_size=1)
        self.bn = nn.BatchNorm2d(mid)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mid, channel, kernel_size=1)
        self.conv_w = nn.Conv2d(mid, channel, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return x * a_h * a_w



# --- DILA: Dual-Interactive Lightweight Attention ---
class DILA(nn.Module):
    """
    Dual-Interactive Lightweight Attention (DILA)
    - Split channels into two paths (A: channel routing, B: spatial refine)
    - A-path: light channel routing with GAP -> small 1x1 projection -> mask
    - B-path: lightweight multi-scale depthwise spatial refinement
    - Cross-injection: A->B and B->A side information for mutual correction
    - Fuse and re-project with a learnable residual scaling alpha
    Intended to be lightweight and parameter-efficient.
    """
    def __init__(self, channels, r=0.5, groups=4, alpha_init=0.1):
        super().__init__()
        C1 = max(4, int(channels * r))
        C2 = channels - C1
        self.C1 = C1
        self.C2 = C2

        # A-path: channel routing (very light)
        self.a_mask_fc = nn.Conv2d(C1, C1, kernel_size=1, bias=True)
        self.a_proj = nn.Conv2d(C1, C1, kernel_size=1, groups=groups, bias=True)
        self.a_gate = nn.Parameter(torch.tensor(0.2))

        # B-path: spatial refine (depthwise multi-scale)
        self.b_dw3 = nn.Conv2d(C2, C2, kernel_size=3, padding=1, groups=C2, bias=True)
        self.b_dw5 = nn.Conv2d(C2, C2, kernel_size=5, padding=2, groups=C2, bias=True)
        self.b_proj = nn.Conv2d(C2, C2, kernel_size=1, bias=True)

        # cross injection projections
        self.a2b = nn.Conv2d(C1, 1, kernel_size=1, bias=True)
        self.b2a = nn.Conv2d(1, C1, kernel_size=1, bias=True)

        # fusion & expand
        self.fuse = nn.Conv2d(C1 + C2, channels, kernel_size=1, bias=True)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: B,C,H,W
        b, c, h, w = x.size()
        x_a = x[:, :self.C1, :, :]
        x_b = x[:, self.C1:, :, :]

        # A-path: global descriptor -> mask logits
        p = F.adaptive_avg_pool2d(x_a, 1)                 # B,C1,1,1
        mask_logits = self.a_mask_fc(p)                   # B,C1,1,1

        # B-path spatial processing
        y3 = self.b_dw3(x_b)
        y5 = self.b_dw5(x_b)
        sp = (y3 + y5)                                   # B,C2,H,W
        sp_proj = self.b_proj(sp)                        # B,C2,H,W

        # B->A injection: produce global bias from sp (avg over channels)
        sp_global = F.adaptive_avg_pool2d(sp_proj, 1)    # B,C2,1,1
        sp_bias = sp_global.mean(dim=1, keepdim=True)    # B,1,1,1
        mask_logits = mask_logits + self.b2a(sp_bias)    # inject

        # finalize mask
        mask = self.sigmoid(mask_logits) * self.a_gate   # B,C1,1,1
        x_a_hat = x_a * mask                             # B,C1,H,W (broadcast)
        x_a_hat = self.a_proj(x_a_hat)

        # A->B injection: global channel bias to spatial path
        a2b_bias = self.a2b(p)                           # B,1,1,1
        sp_map = self.sigmoid(sp_proj + a2b_bias)        # B,C2,H,W
        x_b_hat = x_b * sp_map

        z = torch.cat([x_a_hat, x_b_hat], dim=1)        # B,C,H,W
        out = self.fuse(z)
        return x + self.alpha * out


class ULAttention(nn.Module):
    """
    ULAttention: Unified lightweight attention combining channel and spatial
    attention with a bottleneck. Uses grouped expansion on the channel
    re-projection to reduce parameter cost. Default reduction=16.
    This optimized variant uses ReLU and a lightweight spatial conv (k=3,p=1)
    to reduce FLOPs.
    The att_mode controls which attention branches are enabled:
      - 'ul': both CA and SA
      - 'ca': simple Channel Attention (SE-like)
      - 'cca': Coordinate Channel Attention (default)
      - 'eca': Efficient Channel Attention
      - 'sa': only spatial attention
      - 'esa': Efficient Spatial Attention (default)
      - 'lsa': Lightweight Spatial Attention
    """
    def __init__(self, channel, reduction=16, att_mode='ul'):
        super().__init__()
        self.att_mode = att_mode
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        mid = max(1, channel // reduction)
        restore_groups = 2 if (channel % 2 == 0) else 1
        # --- Channel Attention selection ---
        ca_mode = att_mode
        if ca_mode == 'cca':
            self.ca_conv = CCAAttention(channel, reduction)
        elif ca_mode == 'eca':
            self.ca_conv = ECA(channel)
        elif ca_mode == 'ca':
            self.ca_conv = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, bias=True)
            )
        else:
            self.ca_conv = CCAAttention(channel, reduction)
        if att_mode == 'esa':
            self.sa_conv = ESA(channel)
        elif att_mode == 'lsa':
            self.sa_conv = LSA(channel)
        else:
            self.sa_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.fuse = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))

    def forward(self, x):
        if self.att_mode == 'ca' or self.att_mode == 'cca':
            ca = self.ca_conv(x)                           # (B, C, H, W) for CCA or (B, C, 1, 1) for original CA
            out = x * ca
            return x + out
        elif self.att_mode in ['sa', 'esa', 'lsa']:
            # ESA/LSA 直接接收多通道输入，普通 SA 使用均值单通道
            if isinstance(self.sa_conv, (ESA, LSA)):
                sa = self.sa_conv(x)
            else:
                sa = torch.sigmoid(self.sa_conv(torch.mean(x, dim=1, keepdim=True)))

            # 对齐通道维度
            if sa.size(1) != x.size(1):
                sa_map = F.interpolate(sa, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
                if sa_map.size(1) != x.size(1):
                    repeat_times = x.size(1) // sa_map.size(1)
                    if repeat_times >= 1 and x.size(1) % sa_map.size(1) == 0:
                        sa_map = sa_map.repeat(1, repeat_times, 1, 1)
                    else:
                        sa_map = sa_map.mean(dim=1, keepdim=True).repeat(1, x.size(1), 1, 1)
            else:
                sa_map = sa
            out = x * sa_map
            return x + out
        else:  # 'ul' both CA and SA
            pooled = self.global_pool(x)
            ca = torch.sigmoid(self.ca_conv(pooled))                           # (B, C, 1, 1)
            sa = torch.sigmoid(self.sa_conv(torch.mean(x, dim=1, keepdim=True)))  # (B, 32, H, W) -> broadcast
            # compute normalized fusion weights
            w = torch.softmax(self.fuse, dim=0)
            f1, f2 = w[0], w[1]
            # broadcast CA to spatial shape
            ca_map = ca.expand_as(x)
            # if sa has channel != C, project it to C by simple repeat/interpolation
            if sa.size(1) != x.size(1):
                sa_map = F.interpolate(sa, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
                if sa_map.size(1) != x.size(1):
                    repeat_times = x.size(1) // sa_map.size(1)
                    if repeat_times >= 1 and x.size(1) % sa_map.size(1) == 0:
                        sa_map = sa_map.repeat(1, repeat_times, 1, 1)
                    else:
                        sa_map = sa_map.mean(dim=1, keepdim=True).repeat(1, x.size(1), 1, 1)
            else:
                sa_map = sa
            out = x * (f1 * ca_map + f2 * sa_map)
            return x + out


# NEW: StageULAttention class for stage-level attention
class StageULAttention(nn.Module):
    """
    Wrapper that applies a lightweight 1x1 projection around ULAttention and
    a learnable residual gating (alpha). This shrinks the parameter/FLOPs
    impact while keeping attention global and stable when applied once per
    body output (stage-level attention).
    """
    def __init__(self, channel, reduction=16, alpha=0.25, att_mode='stage'):
        super().__init__()
        self.proj_in = nn.Conv2d(channel, channel, kernel_size=1, bias=True)
        self.att = ULAttention(channel, reduction=reduction, att_mode='ul')
        self.proj_out = nn.Conv2d(channel, channel, kernel_size=1, bias=True)
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))

    def forward(self, x):
        y = self.proj_in(x)
        y = self.att(y)
        y = self.proj_out(y)
        return x + self.alpha * y


class GADWResidualBlock(nn.Module):
    """
    GADWResidualBlock: Residual block using GADWConv and optional attention.
    """
    def __init__(self, channels, kernel_size=3, res_scale=0.1,
                 use_dwconv=True, idx=0, use_attention=True, att_mode='dila'):
        super().__init__()
        self.use_dwconv = use_dwconv
        self.use_attention = use_attention
        self.att_mode = att_mode
        padding = kernel_size // 2

        if use_dwconv:
            self.conv1 = GADWConv(channels)
            self.conv2 = GADWConv(channels)
        else:
            self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=True)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=True)

        # use lightweight ReLU6 to improve numerical stability on low-precision devices
        self.act = nn.ReLU6(inplace=True)

        # learnable residual scaling as a parameter
        self.res_scale = nn.Parameter(torch.tensor(res_scale, dtype=torch.float32))

        # Attention module selection based on att_mode (支持自由组合)
        modes = parse_attention_modes(att_mode)
        self.att_modules = nn.ModuleList()
        if self.use_attention:
            if modes["dila"]:
                self.att_modules.append(DILA(channels, r=0.5, groups=4, alpha_init=0.1))
            if modes["ula"]:
                self.att_modules.append(ULAttention(channels, reduction=16, att_mode="ul"))
            if modes["ca"]:
                self.att_modules.append(ULAttention(channels, reduction=16, att_mode="ca"))
            if modes["cca"]:
                self.att_modules.append(ULAttention(channels, reduction=16, att_mode="cca"))
            if modes["eca"]:
                self.att_modules.append(ULAttention(channels, reduction=16, att_mode="eca"))
            if modes["sa"]:
                self.att_modules.append(ULAttention(channels, reduction=16, att_mode="sa"))
            if modes["esa"]:
                self.att_modules.append(ULAttention(channels, reduction=16, att_mode="esa"))
            if modes["lsa"]:
                self.att_modules.append(ULAttention(channels, reduction=16, att_mode="lsa"))
            if modes["stage"]:
                self.att_modules.append(StageULAttention(channels, reduction=8, alpha=0.25, att_mode="stage"))

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)

        out = out * self.res_scale
        if self.use_attention and len(self.att_modules) > 0:
            for att in self.att_modules:
                out = att(out)
            return x + out
        else:
            # skip attention (no additional attention module here, so just return residual)
            return x + out


class ULRNet(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super().__init__()
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        scale = args.scale[0] if isinstance(args.scale, (list, tuple)) else args.scale

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        m_head = [
            nn.Conv2d(args.n_colors, n_feats, 3, padding=1, bias=True),
            nn.Conv2d(n_feats, n_feats, 1, bias=True),
            nn.ReLU(inplace=True),
        ]

        use_dw = getattr(args, 'use_dwconv', True)
        use_att = getattr(args, 'use_attention', True)
        att_mode = getattr(args, 'att_mode', 'dila')

        m_body = []
        total_blocks = n_resblocks
        for i in range(n_resblocks):
            block = GADWResidualBlock(n_feats, kernel_size=3, res_scale=0.1,
                                      use_dwconv=use_dw, idx=i, use_attention=use_att, att_mode=att_mode)
            block.total_blocks = total_blocks
            m_body.append(block)
        m_body.append(conv(n_feats, n_feats, 3))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        # Remove global_att usage, keep None
        self.global_att = None

        m_tail = [
            nn.Conv2d(n_feats, n_feats, 1, bias=True),
            common.Upsampler(conv, scale, n_feats, act=False),
            nn.Conv2d(n_feats, args.n_colors, 3, padding=1)
        ]

        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        body_layers = list(self.body)
        last_conv = body_layers[-1]
        blocks = body_layers[:-1]

        out = x
        for block in blocks:
            out = block(out)
        res = last_conv(out)

        # self.global_att is None, so skip attention here
        res = res + x
        x = self.tail(res)
        x = self.add_mean(x)
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
                        raise RuntimeError(f"Error copying parameter {name}, model shape {own_state[name].size()}, checkpoint shape {param.size()}")
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError(f'Unexpected key {name} in state_dict')


def make_model(args, parent=False):
    net = ULRNet(args)
    check_modules(net, getattr(args, 'att_mode', None))
    return net


def check_modules(net, att_mode=None):
    """
    检查模型内部模块组成，输出详细注意力结构报告。
    - 识别并显示所有已启用的注意力模块组合（CA、CCA、ECA、SA、ESA、LSA、UL、DILA、Stage）。
    - 解析组合 att_mode（如 'dila+stage', 'cca+esa'），逐个显示模块。
    - 格式整齐，区分主模块（DILA/Stage/UL）和子模块（CA/ESA等）。
    """
    has_gadw = any(isinstance(m, GADWConv) for m in net.modules())
    has_ulatt = any(isinstance(m, ULAttention) for m in net.modules())
    has_stage = any(isinstance(m, StageULAttention) for m in net.modules())
    has_dila = any(isinstance(m, DILA) for m in net.modules())

    # 推断主 attention
    if has_dila:
        att_main = "DILA (Dual-Interactive Lightweight Attention)"
        main_type = "DILA"
        sub_branch = "内含 Channel–Spatial 双交互机制"
    elif has_stage:
        att_main = "StageULAttention (Stage-level ULAttention)"
        main_type = "Stage"
        sub_branch = "ULAttention 封装，轻量1x1投影 + 残差缩放"
    elif has_ulatt:
        att_main = "ULAttention (Unified Lightweight Attention)"
        main_type = "UL"
        # 从第一个 ULAttention 中读取 att_mode
        first_ul = next((m for m in net.modules() if isinstance(m, ULAttention)), None)
        sub_branch = f"分支类型: {getattr(first_ul, 'att_mode', 'unknown').upper()}" if first_ul else "未知"
    else:
        att_main = "无 Attention 模块"
        main_type = None
        sub_branch = "（纯卷积结构）"

    att_mode_str = att_mode if att_mode is not None else 'default'

    # 解析所有组合模式
    def parse_modes(mode_str):
        if not mode_str:
            return []
        return [x.strip().lower() for x in mode_str.replace(',', '+').split('+') if x.strip()]

    # 支持的所有 attention 子模块
    att_submodules = ["ca", "cca", "eca", "sa", "esa", "lsa"]
    att_mainmodules = ["dila", "stage", "ul", "ula"]

    # 解析激活的 attention 组合
    modes_enabled = []
    if att_mode is not None:
        modes_enabled = parse_modes(att_mode)
    else:
        # 尝试从模型中推断
        for m in net.modules():
            if isinstance(m, DILA):
                modes_enabled.append("dila")
            if isinstance(m, StageULAttention):
                modes_enabled.append("stage")
            if isinstance(m, ULAttention):
                attm = getattr(m, "att_mode", None)
                if attm and attm.lower() not in modes_enabled:
                    modes_enabled.append(attm.lower())
    # 去重、排序
    modes_enabled = sorted(set(modes_enabled), key=lambda x: (att_mainmodules + att_submodules).index(x) if x in (att_mainmodules + att_submodules) else 99)

    # 分类主模块和子模块
    main_mods = [m for m in modes_enabled if m in att_mainmodules]
    sub_mods = [m for m in modes_enabled if m in att_submodules]

    # 自动检测模型中实际存在的注意力子模块
    has_eca = any(isinstance(m, ECA) for m in net.modules())
    has_esa = any(isinstance(m, ESA) for m in net.modules())
    has_lsa = any(isinstance(m, LSA) for m in net.modules())
    has_cca = any(isinstance(m, CCAAttention) for m in net.modules())

    auto_submods = []
    if has_cca: auto_submods.append("cca")
    if has_eca: auto_submods.append("eca")
    if has_esa: auto_submods.append("esa")
    if has_lsa: auto_submods.append("lsa")

    # 如果模型中存在 StageULAttention 或 ULAttention(att_mode='ul')，则自动加入 CCA 子模块
    for m in net.modules():
        if isinstance(m, StageULAttention) or (isinstance(m, ULAttention) and getattr(m, "att_mode", "") == "ul"):
            if "cca" not in auto_submods:
                auto_submods.append("cca")

    # 合并命令行/推断与实际模型检测到的 attention 子模块
    sub_mods = sorted(set(sub_mods + auto_submods))

    # 打印结构报告
    print("========== [模型结构报告] ==========")
    print(f"[主干卷积] GADWConv: {has_gadw}")
    print(f"[注意力类型] {att_main}")
    print(f"[分支说明] {sub_branch}")
    print(f"[命令行指定模式] --att_mode {att_mode_str}")
    print("-" * 38)
    print("[Attention 主模块]:", "、".join([m.upper() for m in main_mods]) if main_mods else "无")
    print("[Attention 子模块]:", "、".join([m.upper() for m in sub_mods]) if sub_mods else "无")
    # 打印实际模型中启用的 attention module 类名
    active_modules = [m.__class__.__name__ for m in net.modules() if isinstance(m, (ULAttention, StageULAttention, DILA))]
    print(f"[模型内启用的注意力结构]: {', '.join(sorted(set(active_modules)))}")
    print("=====================================")