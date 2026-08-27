import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, DeepGraphInfomax

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return x

def corruption(x, edge_index):
    permuted_x = x[torch.randperm(x.size(0))]
    return permuted_x, edge_index

class DGI(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, num_layers=1, dropout=False, use_pred=False):
        super(DGI, self).__init__()
        self.hidden_channels = hidden_channels
        self.encoder = Encoder(in_channels, hidden_channels)
        self.dgi = DeepGraphInfomax(
            hidden_channels=hidden_channels,
            encoder=self.encoder,
            summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=corruption
        )

    def forward(self, x, edge_index):
        pos_z, neg_z, summary = self.dgi(x, edge_index)
        return (pos_z, neg_z, summary)