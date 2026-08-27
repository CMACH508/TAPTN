"""
Reference: https://github.com/XiaoxinHe/TAPE/blob/main/core/data_utils/load_arxiv.py
"""

try:
    from ogb.nodeproppred import PygNodePropPredDataset
except ImportError:
    PygNodePropPredDataset = None

try:
    import torch_geometric.transforms as T
except ImportError:
    T = None
    torch = None

import pandas as pd
import random
import numpy as np
from collections import Counter


def get_raw_text_arxiv(use_text=False, seed=0):
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    data = dataset[0]

    idx_splits = dataset.get_idx_split()
    train_mask = torch.zeros(data.num_nodes).bool()
    val_mask = torch.zeros(data.num_nodes).bool()
    test_mask = torch.zeros(data.num_nodes).bool()
    train_mask[idx_splits['train']] = True
    val_mask[idx_splits['valid']] = True
    test_mask[idx_splits['test']] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # data.edge_index = data.adj_t.to_symmetric()
    data.edge_index
    if not use_text:
        return data, None

    nodeidx2paperid = pd.read_csv(
        'dataset/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz', compression='gzip')
    
    raw_text = pd.read_csv('dataset/ogbn_arxiv/titleabs.tsv', sep='\t')
    raw_text.columns = ['paper id', 'title', 'abs']

    df = pd.merge(nodeidx2paperid, raw_text, on='paper id')

    text = {'title': [], 'abs': [], 'label': []}

    for ti, ab in zip(df['title'], df['abs']):
        text['title'].append(ti)
        text['abs'].append(ab)
    
    # Load the label index to arXiv category mapping data
    label_mapping_data = pd.read_csv('dataset/ogbn_arxiv/mapping/labelidx2arxivcategeory.csv.gz')
    label_mapping_data.columns = ['label_idx', 'arxiv_category']

    for i in range(len(data.y)):
        row = label_mapping_data.loc[label_mapping_data['label_idx'].isin(data.y[i].numpy())]
        # If the row doesn't exist, return a message indicating this
        if len(row) == 0:
            raise 'No matching arXiv category found for this label index.'
    
        # Parse the arXiv category string to be in the desired format 'cs.XX'
        arxiv_category = 'cs.' + row['arxiv_category'].values[0].split()[-1].upper()
        text['label'].append(arxiv_category)
    
    labels = text['label']
    train_id = [i for i in range(len(labels)) if data.train_mask[i]]
    val_id = [i for i in range(len(labels)) if data.val_mask[i]]
    test_id = [i for i in range(len(labels)) if data.test_mask[i]]
    edge_index = data.edge_index
    target_labels = ['cs.GT','cs.MA','cs.RO','cs.NE','cs.IR','cs.SI','cs.CY']
    
    # Find the indices of nodes with the target labels
    target_nodes = [i for i, label in enumerate(labels) if label in target_labels]

    # Step 1: Count the number of nodes in each category
    category_counts = Counter([labels[node] for node in target_nodes])

    # Step 2: Determine the smallest category size
    min_category_size = int(min(category_counts.values())*0.6)
    
    # Step 3: Sample n nodes from each category
    sampled_nodes = []
    for category in target_labels:
        category_nodes = [i for i, label in enumerate(labels) if label == category]
        sampled_nodes.extend(random.sample(category_nodes, min_category_size))

    target_nodes = sampled_nodes
    target_nodes_set = set(target_nodes)

    # Filter train_id, val_id, and test_id to only include nodes in target_nodes
    train_id = [node for node in train_id if node in target_nodes]
    val_id = [node for node in val_id if node in target_nodes]
    test_id = [node for node in test_id if node in target_nodes]
    # other_id = [node for node in target_nodes if node not in train_id + val_id + test_id]
    # random.shuffle(other_id)
    # train_id.extend(other_id[:int(len(other_id)*0.65)])
    # test_id.extend(other_id[int(len(other_id)*0.65):int(len(other_id)*0.8)])
    # val_id.extend(other_id[int(len(other_id)*0.8):])

    # Filter edge_index to only include edges where both nodes are in target_nodes
    filtered_edge_index = []
    for i in range(edge_index.size(1)):
        if edge_index[0, i].item() in target_nodes_set and edge_index[1, i].item() in target_nodes_set:
            filtered_edge_index.append(edge_index[:, i].unsqueeze(1))

    if filtered_edge_index:
        edge_index = torch.cat(filtered_edge_index, dim=1)
    from torch_geometric.utils import degree
    num_nodes = len(labels)
    out_degree = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.long)
    in_degree = degree(edge_index[1], num_nodes=num_nodes, dtype=torch.long)
    train_id = [node for node in train_id if in_degree[node] > 0 or out_degree[node] > 0]
    val_id = [node for node in val_id if in_degree[node] > 0 or out_degree[node] > 0]
    test_id = [node for node in test_id if in_degree[node] > 0 or out_degree[node] > 0]
    # from collections import Counter
   # Calculate the number of nodes per class in the current test set
    test_labels = [labels[node] for node in test_id]
    test_label_counts = Counter(test_labels)
    for label in target_labels:
        if label not in test_label_counts:
            test_label_counts[label] = 0
    # Determine the number of nodes needed per class to balance the test set
    max_count = max(test_label_counts.values())
    needed_per_class = {label: int(max_count/2)  - count for label, count in test_label_counts.items()}

    # Sample nodes from the validation or training set to meet the required number of nodes per class
    additional_test_nodes = []
    for label, needed in needed_per_class.items():
        if needed > 0:
    # Get nodes from the validation and training set with the target label
            candidate_nodes = [node for node in val_id + train_id if labels[node] == label]
            sampled_nodes = random.sample(candidate_nodes, min(needed, len(candidate_nodes)))
            additional_test_nodes.extend(sampled_nodes)
    # Update the test set and masks
    test_id.extend(additional_test_nodes)
    #test_mask = torch.tensor([x in test_id for x in range(num_nodes)])

    # Remove the sampled nodes from the validation and training sets
    val_id = [node for node in val_id if node not in additional_test_nodes]
    train_id = [node for node in train_id if node not in additional_test_nodes]
    t_v_id = train_id + val_id
    random.shuffle(t_v_id)
    test_id = t_v_id[:int(len(t_v_id)*0.5)]
    val_id = t_v_id[int(len(t_v_id)*0.5):]
    data.train_id = train_id
    data.val_id = val_id
    data.test_id = test_id
    data.train_mask = torch.tensor([x in train_id for x in range(num_nodes)])
    data.val_mask = torch.tensor([x in val_id for x in range(num_nodes)])
    data.test_mask = torch.tensor([x in test_id for x in range(num_nodes)])
    data.in_degree = in_degree
    data.out_degree = out_degree
    data.edge_index = edge_index

    return data, text


def generate_arxiv_keys_list():
    label_mapping_data = pd.read_csv('dataset/ogbn_arxiv/mapping/labelidx2arxivcategeory.csv.gz', compression='gzip')
    label_mapping_data.columns = ['label_idx', 'arxiv_category']
    arxiv_categories = label_mapping_data['arxiv_category'].unique()
    return ['cs.' + category.split()[-1].upper() for category in arxiv_categories]

#get_raw_text_arxiv(use_text=True, seed=0)
