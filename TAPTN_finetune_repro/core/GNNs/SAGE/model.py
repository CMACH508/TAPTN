import torch
import torch.nn.functional as F
import csv
from torch_geometric.nn import SAGEConv

def load_gpt_preds_weight(dataset, topk):
    using_pkl=False
    if using_pkl:
        import pickle
        fn = f'gpt_preds/{dataset}.pkl'
        print(f"Loading topk preds from {fn}")
        return pickle.load(open(fn, 'rb'))
    preds = []
    fn = f'gpt_preds/{dataset}_weight.csv'
    print(f"Loading topk preds from {fn}")
    with open(fn, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            inner_list = []
            for value in row:
                inner_list.append(float(value))
            preds.append(inner_list)

    pl = torch.ones(len(preds), topk, dtype=torch.long)
    for i, pred in enumerate(preds):
        pl[i][:len(pred)] = torch.tensor(pred[:topk], dtype=torch.long)+1
    return pl

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, use_pred):
        super(SAGE, self).__init__()
        self.use_pred = use_pred
        if self.use_pred:
            self.encoder = torch.nn.Embedding(out_channels+1, hidden_channels)
            #self.encoder = torch.nn.Linear(out_channels, in_channels)
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        if self.use_pred:
            self.option_weight = load_gpt_preds_weight('cora', 5).view(-1, 5, 1)

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        if self.use_pred:
            x = self.encoder(x)
            #x = self.encoder(x)*self.option_weight.to(x.device)
            x = torch.flatten(x, start_dim=1)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x
