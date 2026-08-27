import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch
import torch.nn.functional as F
from torch_geometric.datasets import WebKB
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeFeatures
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'


class ACMConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(ACMConv, self).__init__(aggr='mean')  # Use 'mean' aggregation
        
        # Define the weight matrices for aggregation, diversification, and identity channels
        self.lin_agg = nn.Linear(in_channels, out_channels)
        self.lin_div = nn.Linear(in_channels, out_channels)
        self.lin_id = nn.Linear(in_channels, out_channels)
        
        # Define the mixing weights computation
        self.lin_mix = nn.Linear(in_channels, 3)  # 3 channels: agg, div, id

    def forward(self, x, edge_index):
        # Add self-loops to the edge index to account for self-node information
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Compute the aggregation channel
        h_agg = self.propagate(edge_index, x=x)
        h_agg = self.lin_agg(h_agg)
        
        # Compute the diversification channel
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype).view(-1, 1)
        h_div = self.lin_div(x - torch.index_add(torch.zeros_like(x), 0, row, x[col]) / deg)
        
        # Compute the identity channel
        h_id = self.lin_id(x)
        
        # Calculate adaptive mixing weights
        mix_weights = F.softmax(self.lin_mix(x), dim=-1)
        alpha, beta, gamma = mix_weights[:, 0], mix_weights[:, 1], mix_weights[:, 2]
        
        # Combine the channels
        out = alpha.unsqueeze(-1) * h_agg + beta.unsqueeze(-1) * h_div + gamma.unsqueeze(-1) * h_id
        return out

    def message(self, x_j):
        return x_j

class ACMGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=False, use_pred=False):
        super(ACMGNN, self).__init__()
        self.conv1 = ACMConv(in_channels, hidden_channels)
        self.conv2 = ACMConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
if __name__ == '__main__':
    def test(model, data):
        model.eval()
        logits = model(data.x, data.edge_index)
        masks = [('Train', data.train_mask), ('Val', data.val_mask), ('Test', data.test_mask)]
        accs = []
        for _, mask in masks:
            correct = logits[mask].argmax(dim=1).eq(data.y[mask]).sum().item()
            accs.append(correct / mask.sum().item())
        return accs
    def train(model, optimizer, data):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        
        #out = F.log_softmax(out, dim=-1)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    dataset = WebKB(root='.', name='Cornell', transform=NormalizeFeatures())
    data = dataset[0].to(device)
    data.train_mask = data.train_mask[:, 0] 
    data.val_mask = data.val_mask[:, 0]
    data.test_mask = data.test_mask[:, 0]
    # Adjust in_channels, hidden_channels, out_channels to match data
    model = ACMGNN(
        in_channels=dataset.num_features,
        hidden_channels=128,
        out_channels=dataset.num_classes
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, 401):
        loss = train(model, optimizer, data)
        train_acc, val_acc, test_acc = test(model, data)
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {loss:.4f}, Train: {train_acc:.4f}, "
                  f"Val: {val_acc:.4f}, Test: {test_acc:.4f}")
