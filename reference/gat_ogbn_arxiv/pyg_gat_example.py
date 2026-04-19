# 这个文件更像 ogbn-arxiv 的结构对照模板，用来和当前 formal 主线的形状配置对位。
# 真正的训练入口在 scripts/train_ogbn_arxiv_gat.py。
"""
Reference GAT template for ogbn-arxiv.

This file is retained as a read-only architecture reference that documents the
PyG GAT shape contract used by the project mainline. The executable training
entry lives in scripts/train_ogbn_arxiv_gat.py.

Contract:
- 2-layer GAT
- input dim = 128
- hidden = 8 heads x 8 dim
- output heads = 1
- class count = 40
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(self, in_channels: int = 128, hidden_channels: int = 8,
                 out_channels: int = 40, heads: int = 8):
        super().__init__()
        # 隐层保持多头拼接，对应项目主线里的隐藏层 concat 语义。
        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            concat=True,
            dropout=0.6,
        )
        # 输出层固定为单头不拼接，对应最终 logits 头的平均退化情形。
        self.conv2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=0.6,
        )

    def forward(self, x, edge_index):
        # 先做输入 dropout，再进入隐藏层多头 attention。
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        # 输出层不再做多头拼接，最后直接给出按类的对数概率。
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)