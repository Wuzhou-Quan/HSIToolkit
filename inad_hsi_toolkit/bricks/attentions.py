import torch


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, torch.nn.Conv2d):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)


class ChannelAttention(torch.nn.Module):
    """Channel attention module.

    This module performs channel-wise attention by computing attention weights based on the input feature map.
    Only output the attention response itself.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int, optional): Number of output channels. If not specified, defaults to `in_channels`.
        inter_channels (int, optional): Number of intermediate channels used for computations.
            If not specified, defaults to `in_channels` divided by 16.
    """

    def __init__(self, in_channels: int, out_channels: int = None, inter_channels: int = None):
        super().__init__()
        out_channels = out_channels or in_channels
        inter_channels = inter_channels or in_channels // 16
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.proj1 = torch.nn.Conv2d(in_channels, inter_channels, 1, padding=0)
        self.act1 = torch.nn.ReLU()
        self.proj2 = torch.nn.Conv2d(inter_channels, in_channels, 1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avgpool(x)
        y = self.proj1(y)
        y = self.act1(y)
        y = self.proj2(y)
        y = y.sigmoid()
        return y


def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    temp = query.bmm(key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    softmax = torch.nn.functional.softmax(temp / scale, dim=-1)
    return softmax.bmm(value)
    # temp = torch.einsum("bqd,bkd->bqk", query, key)
    # scale = query.size(-1) ** 0.5
    # attention_weights = F.softmax(temp / scale, dim=-1)
    # return torch.einsum("bqk,bkv->bqv", attention_weights, value)


class HeadAttention(torch.nn.Module):
    def __init__(self, dim_in: int, dim_k: int, dim_v: int):
        super().__init__()
        self.q = torch.nn.Linear(dim_in, dim_k)
        self.k = torch.nn.Linear(dim_in, dim_k)
        self.v = torch.nn.Linear(dim_in, dim_v)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))


class CrossAttention(torch.nn.Module):
    def __init__(self, qv_in_channels, k_in_channels):
        super(CrossAttention, self).__init__()
        self.conv_query = torch.nn.Conv2d(in_channels=qv_in_channels, out_channels=qv_in_channels, kernel_size=1, bias=True)
        self.conv_key = torch.nn.Conv2d(in_channels=k_in_channels, out_channels=k_in_channels, kernel_size=1, bias=True)
        self.conv_value = torch.nn.Conv2d(in_channels=qv_in_channels, out_channels=k_in_channels, kernel_size=1, bias=True)
        self.conv_out = torch.nn.Sequential(
            torch.nn.BatchNorm2d(qv_in_channels, eps=0.0001, momentum=0.95),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(in_channels=qv_in_channels, out_channels=qv_in_channels, kernel_size=1, bias=True),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(in_channels=qv_in_channels, out_channels=qv_in_channels, kernel_size=1, bias=True),
        )
        fill_fc_weights(self)

    def forward(self, q, v):
        query = self.conv_query(q)
        key = self.conv_key(v)
        value = self.conv_value(q)

        query = query.view(query.size(0), query.size(1), -1)
        key = key.view(key.size(0), key.size(1), -1)
        value = value.view(value.size(0), value.size(1), -1)

        out = scaled_dot_product_attention(query, key, value)

        out = out.permute(0, 2, 1).contiguous()
        out = out.view(q.size(0), -1, q.size(2), q.size(3))
        out = self.conv_out(out)

        return out


class MultiHeadCrossAttention(torch.nn.Module):
    """Multi-Head Attention"""

    def __init__(self, n_heads, q_in_channels, kv_in_channels):
        super().__init__()

        self.n_head = n_heads
        self.conv_query = torch.nn.Conv2d(in_channels=q_in_channels, out_channels=q_in_channels * n_heads, kernel_size=1, bias=True)
        self.conv_key = torch.nn.Conv2d(in_channels=kv_in_channels, out_channels=kv_in_channels * n_heads, kernel_size=1, bias=True)
        self.conv_value = torch.nn.Conv2d(in_channels=kv_in_channels, out_channels=kv_in_channels * n_heads, kernel_size=1, bias=True)

        self.conv_out = torch.nn.Sequential(
            torch.nn.BatchNorm2d(kv_in_channels * n_heads, eps=0.0001, momentum=0.95),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(in_channels=kv_in_channels * n_heads, out_channels=kv_in_channels * n_heads, kernel_size=1, bias=True),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(in_channels=kv_in_channels * n_heads, out_channels=kv_in_channels, kernel_size=1, bias=True),
        )

    def forward(self, q, kv):
        query = self.conv_query(q)
        key = self.conv_key(kv)
        value = self.conv_value(kv)

        query = query.view(query.size(0), query.size(1), -1)
        key = key.view(key.size(0), key.size(1), -1)
        value = value.view(value.size(0), value.size(1), -1)

        out = scaled_dot_product_attention(query, key, value)

        out = out.permute(0, 2, 1).contiguous()
        out = out.view(kv.size(0), -1, kv.size(2), kv.size(3))
        out = self.conv_out(out)

        return out
