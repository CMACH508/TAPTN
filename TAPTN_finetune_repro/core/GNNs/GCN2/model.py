import torch
import torch.nn.functional as F


from torch_geometric.nn import GCN2Conv


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, use_pred):
        super(GCN, self).__init__()
        self.use_pred = use_pred
        if self.use_pred:
            self.encoder = torch.nn.Embedding(out_channels+1, hidden_channels)
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCN2Conv(in_channels, in_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(in_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCN2Conv(in_channels, in_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(in_channels))
        self.convs.append(GCN2Conv(in_channels, in_channels, cached=True))
        self.out_lin = torch.nn.Linear(in_channels, out_channels)

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        if self.use_pred:
            x = self.encoder(x)
            x = torch.flatten(x, start_dim=1)
        x0=x
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, x0, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, x0, adj_t)
        x = self.out_lin(x)
        # return x.log_softmax(dim=-1)
        return x
