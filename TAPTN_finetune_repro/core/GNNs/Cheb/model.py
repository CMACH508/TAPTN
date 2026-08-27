import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

class ChebNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=True, use_pred=False):
        super(ChebNet, self).__init__()
        self.convs = torch.nn.ModuleList()
        K = 3  # Polynomial order for Chebyshev convolution
        # Input layer
        self.convs.append(ChebConv(in_channels, hidden_channels, K))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(ChebConv(hidden_channels, hidden_channels, K))
        # Output layer
        self.convs.append(ChebConv(hidden_channels, out_channels, K))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x