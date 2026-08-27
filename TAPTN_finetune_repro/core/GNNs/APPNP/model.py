import torch
import torch.nn.functional as F
from torch_geometric.nn import APPNP

class APPNPModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1, dropout=False, use_pred=False, K=10, alpha=0.1):
        super(APPNPModel, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
        self.prop = APPNP(K=K, alpha=alpha)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, edge_index)
        return x