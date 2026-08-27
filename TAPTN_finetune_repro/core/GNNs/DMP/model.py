import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

class DMPConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(DMPConv, self).__init__(aggr='add')  # Use 'add' aggregation.
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Weight matrices.
        self.W = torch.nn.Linear(in_channels, out_channels)
        self.W_c = torch.nn.Linear(2 * in_channels, in_channels)
        self.W_bar_c = torch.nn.Linear(2 * in_channels, in_channels)

    def forward(self, x, edge_index):
        # x: Node features [N, F].
        # edge_index: Graph connectivity [2, E].

        # Compute aggregated neighbor features (\bar{h}_v) without self-loops.
        agg_out = self.propagate(edge_index, x=x)  # [N, F]

        # Compute c_v and \bar{c}_v.
        h = x  # [N, F]
        h_bar = agg_out  # [N, F]
        concat = torch.cat([h, h_bar], dim=1)  # [N, 2F]

        c_v = torch.tanh(self.W_c(concat))  # [N, F]
        bar_c_v = torch.tanh(self.W_bar_c(concat))  # [N, F]

        # Compute element-wise products.
        temp1 = c_v * h  # [N, F]
        temp2 = bar_c_v * h_bar  # [N, F]

        # Combine and apply linear transformation and activation.
        out = temp1 + temp2  # [N, F]
        out = self.W(out)  # [N, out_channels]
        out = torch.relu(out)  # Apply activation function.

        return out

    def message(self, x_j):
        return x_j  # Send neighbor features.

    # The 'aggregate' method defaults to 'add' and does not need to be overridden.
    # The 'update' method does not require overriding unless additional computation is needed.


class DMP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=False, use_pred=False):
        super(DMP, self).__init__()
        self.conv1 = DMPConv(in_channels, hidden_channels)
        self.conv2 = DMPConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        #x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
