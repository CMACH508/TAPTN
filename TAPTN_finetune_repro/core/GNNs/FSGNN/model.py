import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class FSGNN(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1, dropout=False, use_pred=False, num_hops=2):
        super(FSGNN, self).__init__()
        
        # Define linear transformation layers for each hop
        self.linears = torch.nn.ModuleList([
            Linear(in_channels, hidden_channels) for _ in range(num_hops * 2 + 1)
        ])
        
        # Scalar parameters for feature weighting (soft-selection)
        self.alphas = Parameter(torch.ones(num_hops * 2 + 1))
        
        # Output linear layer
        self.out_linear = Linear(hidden_channels * (num_hops * 2 + 1), out_channels)
        
        # Number of hops to aggregate features
        self.num_hops = num_hops

    def forward(self, x, edge_index):
        # Compute different hop features
        x_hops = []
        
        # Initial feature (node's own feature)
        x_hops.append(x)
        
        # Adding hop features without self-loops (no-loop aggregation)
        for k in range(1, self.num_hops + 1):
            x = self.propagate(edge_index, x=x)
            x_hops.append(x)
        
        # Reset initial x and reintroduce self-loops
        x = x_hops[0]
        edge_index_with_loops = add_self_loops(edge_index)[0]
        
        # Adding hop features with self-loops
        for k in range(1, self.num_hops + 1):
            x = self.propagate(edge_index_with_loops, x=x)
            x_hops.append(x)
        
        # Apply unique transformations to each hop feature
        x_transformed = [linear(x_h) for x_h, linear in zip(x_hops, self.linears)]
        
        # Softmax normalization over weights (soft-selection mechanism)
        alphas_normalized = F.softmax(self.alphas, dim=0)
        
        # Apply weighting and concatenate
        x_out = torch.cat([alpha * x_h for alpha, x_h in zip(alphas_normalized, x_transformed)], dim=1)
        
        # L2 normalization
        x_out = F.normalize(x_out, p=2, dim=1)
        
        # Final layer
        x_out = self.out_linear(x_out)
        return x_out
    
    def propagate(self, edge_index, x):
        # Aggregate features by summing over neighbors
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # Normalize with degree and sum over edges
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        out = torch.zeros_like(x)
        out.index_add_(0, row, norm.view(-1, 1) * x[col])
        return out


