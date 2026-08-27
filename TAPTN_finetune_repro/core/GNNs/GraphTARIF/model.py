"""
GraphTARIF: Linear Graph Transformer with Augmented Rank and Improved Focus
(WWW 2026, arXiv:2510.10631)

落地实现，契合本仓库 GNN 接口：构造签名 (in_channels, hidden_channels,
out_channels, num_layers, dropout, use_pred)，forward(x, edge_index) → logits，
全图监督节点分类。

核心组件（忠实于论文）：
  · 式(12) 高 rank 线性注意力：Z = φ(Q)(φ(K)ᵀV) + λ·σ(a)·GAT(V)
        - 全局线性注意力（核特征图 φ，O(N) 复杂度）提供长程上下文；
        - 门控 GAT 局部分支提升等效注意力矩阵的秩；σ(a) 为可学习门控，λ 为缩放。
  · 式(14) 可学习 log-power 锐化：f(x;p,q)=x·(log(1+x^p))^q，
        p=1+α·σ(w_p), q=1+β·σ(w_q)；作用于 φ(Q)、φ(K) 上降低注意力分布熵。
  · 式(13) 节点级后调制：Z̄ = ψ(X) ⊙ Z，ψ 为线性投影，保留节点自身信息、提升秩与判别性。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


def _phi(x):
    """正核特征图（保证注意力分值为正，是 log-power 的合法输入）。"""
    return F.elu(x) + 1.0


class GraphTARIFLayer(nn.Module):
    def __init__(self, dim, dropout, alpha=1.0, beta=1.0, lam=1.0):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.psi = nn.Linear(dim, dim)                     # 式(13) 节点级后调制
        self.gat = GATConv(dim, dim, heads=1, concat=False)  # 式(12) 局部分支
        self.gate_a = nn.Parameter(torch.zeros(1))         # σ(a) 可学习门控标量
        self.wp = nn.Parameter(torch.zeros(1))             # log-power p
        self.wq = nn.Parameter(torch.zeros(1))             # log-power q
        self.alpha, self.beta, self.lam = alpha, beta, lam
        self.norm = nn.LayerNorm(dim)
        self.dropout = dropout

    def _logpow(self, x):
        """式(14) f(x;p,q)=x·(log(1+x^p))^q，x>0；p,q∈(1,1+·)保证单调凸、稳定。"""
        p = 1.0 + self.alpha * torch.sigmoid(self.wp)
        q = 1.0 + self.beta * torch.sigmoid(self.wq)
        return x * torch.log1p(x.pow(p)).clamp(min=1e-9).pow(q)

    def forward(self, x, edge_index):
        h = self.norm(x)
        Q = self._logpow(_phi(self.q(h)))                  # [N,d]，已锐化
        K = self._logpow(_phi(self.k(h)))                  # [N,d]
        V = self.v(h)                                      # [N,d]
        # 全局线性注意力（带分母归一化，数值稳定）：O(N·d²)
        KV = K.transpose(0, 1) @ V                         # [d,d]
        num = Q @ KV                                       # [N,d]
        den = (Q * K.sum(dim=0, keepdim=True)).sum(dim=-1, keepdim=True).clamp(min=1e-6)
        z_global = num / den                               # [N,d]
        z_local = self.gat(V, edge_index)                  # [N,d]
        z = z_global + self.lam * torch.sigmoid(self.gate_a) * z_local
        z = self.psi(x) * z                                # 节点级后调制（逐元素）
        z = F.dropout(F.relu(z), p=self.dropout, training=self.training)
        return x + z                                       # 残差


class GraphTARIF(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers, dropout, use_pred, topk=5):
        super().__init__()
        self.use_pred = use_pred
        if use_pred:
            # gpt topk 预测：整数 id → 嵌入后展平（in_channels 已 = hidden*topk）
            self.encoder = nn.Embedding(out_channels + 1, hidden_channels)
            self.topk = max(1, in_channels // hidden_channels)
            fc_in = hidden_channels * self.topk
        else:
            fc_in = in_channels
        self.input_fc = nn.Linear(fc_in, hidden_channels)
        self.layers = nn.ModuleList(
            [GraphTARIFLayer(hidden_channels, dropout) for _ in range(max(1, num_layers))])
        self.out_fc = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def reset_parameters(self):
        for m in self.modules():
            if hasattr(m, 'reset_parameters') and m is not self:
                m.reset_parameters()

    def forward(self, x, adj_t):
        # adj_t 实为 edge_index（与本仓库其它模型一致）
        if self.use_pred:
            x = self.encoder(x)
            x = torch.flatten(x, start_dim=1)
        x = self.input_fc(x)
        x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        for layer in self.layers:
            x = layer(x, adj_t)
        return self.out_fc(x)
