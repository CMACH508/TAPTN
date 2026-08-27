
import dgl
import torch
from torch.utils.data import Dataset as TorchDataset

# convert PyG dataset to DGL dataset


class CustomDGLDataset(TorchDataset):
    def __init__(self, name, pyg_data):
        self.name = name
        self.pyg_data = pyg_data

    def __len__(self):
        return 1

    def __getitem__(self, idx):

        data = self.pyg_data
        g = dgl.DGLGraph()
        edge_index = data.edge_index
        g.add_nodes(data.num_nodes)
        g.add_edges(edge_index[0], edge_index[1])

        if data.edge_attr is not None:
            g.edata['feat'] = torch.FloatTensor(data.edge_attr)
        if data.x is not None:
            g.ndata['feat'] = torch.FloatTensor(data.x)
        g.ndata['label'] = torch.LongTensor(data.y)
        return g

    @property
    def train_mask(self):
        return self.pyg_data.train_mask

    @property
    def val_mask(self):
        return self.pyg_data.val_mask

    @property
    def test_mask(self):
        return self.pyg_data.test_mask


# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None, llm_label_idx=None,
                 gpt_soft_label=None):
        self.encodings = encodings
        self.labels = labels
        # llm_label_idx: list/Tensor[int]，每个节点的 LLM 预测类别索引（-1=未知）。
        # 由 lm_trainer 从 data.llm_label_idx 传入；None 表示不使用此特征。
        if llm_label_idx is not None:
            if isinstance(llm_label_idx, torch.Tensor):
                self.llm_label_idx = llm_label_idx.tolist()
            else:
                self.llm_label_idx = list(llm_label_idx)
        else:
            self.llm_label_idx = None

        # gpt_soft_label: FloatTensor(N, n_classes) — cora 方案C 软独热分布。
        # 由 lm_trainer 从 data.gpt_soft_label 传入；None 表示不使用。
        if gpt_soft_label is not None:
            if isinstance(gpt_soft_label, torch.Tensor):
                self.gpt_soft_label = gpt_soft_label   # 保留为 Tensor，__getitem__ 按行索引
            else:
                self.gpt_soft_label = torch.tensor(gpt_soft_label, dtype=torch.float32)
        else:
            self.gpt_soft_label = None

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['node_id'] = idx
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        if self.llm_label_idx is not None:
            item["llm_label_idx"] = torch.tensor(self.llm_label_idx[idx],
                                                  dtype=torch.long)
        if self.gpt_soft_label is not None:
            item["gpt_soft_label"] = self.gpt_soft_label[idx]   # FloatTensor(n_classes,)
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])
