import os
import torch
import pandas as pd
from torch_geometric.data import Data

def get_raw_text_arxiv_2023(use_text=True, base_path=None):
    if base_path is None:
        _a = os.environ.get('TAPTN_ASSETS')
        base_path = os.path.join(_a, 'dataset/arxiv_2023') if _a else 'dataset/arxiv_2023'
    # Load processed data
    edge_index = torch.load(os.path.join(base_path, "processed", "edge_index.pt"))
    
    # Load raw data
    # edge_df = pd.read_csv(os.path.join(base_path, "raw", "edge.csv.gz"), compression='gzip')
    titles_df = pd.read_csv(os.path.join(base_path, "raw", "titles.csv.gz"), compression='gzip')
    abstracts_df = pd.read_csv(os.path.join(base_path, "raw", "abstracts.csv.gz"), compression='gzip')
    ids_df = pd.read_csv(os.path.join(base_path, "raw", "ids.csv.gz"), compression='gzip')
    labels_df = pd.read_csv(os.path.join(base_path, "raw", "labels.csv.gz"), compression='gzip')
    
    # Load split data
    train_id_df = pd.read_csv(os.path.join(base_path, "split", "train.csv.gz"), compression='gzip')
    val_id_df = pd.read_csv(os.path.join(base_path, "split", "valid.csv.gz"), compression='gzip')
    test_id_df = pd.read_csv(os.path.join(base_path, "split", "test.csv.gz"), compression='gzip')
    
    num_nodes = len(ids_df)
    titles = titles_df['titles'].tolist()
    abstracts = abstracts_df['abstracts'].tolist()
    ids = ids_df['ids'].tolist()
    labels = labels_df['labels'].tolist()
    train_id = train_id_df['train_id'].tolist()
    val_id = val_id_df['val_id'].tolist()
    test_id = test_id_df['test_id'].tolist()
    
    features = torch.load(os.path.join(base_path, "processed", "features.pt"))

    y = torch.load(os.path.join(base_path, "processed", "labels.pt"))
    # Define the target labels
    target_labels = ['cs.GT','cs.MA','cs.RO','cs.NE','cs.IR','cs.SI','cs.CY']
    # Find the indices of nodes with the target labels
    target_nodes = [i for i, label in enumerate(labels) if label in target_labels]
    target_nodes_set = set(target_nodes)
    # Filter train_id, val_id, and test_id to only include nodes in target_nodes
    train_id = [node for node in train_id if node in target_nodes]
    val_id = [node for node in val_id if node in target_nodes]
    test_id = [node for node in test_id if node in target_nodes]
    
    # Filter edge_index to only include edges where both nodes are in target_nodes
    filtered_edge_index = []
    for i in range(edge_index.size(1)):
        if edge_index[0, i].item() in target_nodes_set and edge_index[1, i].item() in target_nodes_set:
            filtered_edge_index.append(edge_index[:, i].unsqueeze(1))

    if filtered_edge_index:
        edge_index = torch.cat(filtered_edge_index, dim=1)
    from torch_geometric.utils import degree
    num_nodes = features.size(0)
    out_degree = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.long)
    in_degree = degree(edge_index[1], num_nodes=num_nodes, dtype=torch.long)
    train_id = [node for node in train_id if in_degree[node] > 0 or out_degree[node] > 0]
    val_id = [node for node in val_id if in_degree[node] > 0 or out_degree[node] > 0]
    test_id = [node for node in test_id if in_degree[node] > 0 or out_degree[node] > 0]
    from collections import Counter
    import random
   # Calculate the number of nodes per class in the current test set
    test_labels = [labels[node] for node in test_id]
    test_label_counts = Counter(test_labels)
    for label in target_labels:
        if label not in test_label_counts:
            test_label_counts[label] = 0
    # Determine the number of nodes needed per class to balance the test set
    max_count = max(test_label_counts.values())
    needed_per_class = {label: max_count - count + 20 for label, count in test_label_counts.items()}

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
    test_mask = torch.tensor([x in test_id for x in range(num_nodes)])

    # Remove the sampled nodes from the validation and training sets
    val_id = [node for node in val_id if node not in additional_test_nodes]
    train_id = [node for node in train_id if node not in additional_test_nodes]
 

    train_mask = torch.tensor([x in train_id for x in range(num_nodes)])
    val_mask = torch.tensor([x in val_id for x in range(num_nodes)])
    test_mask = torch.tensor([x in test_id for x in range(num_nodes)])
    
    data = Data(
        x=features,
        y=y,
        paper_id=ids,
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=num_nodes,
    )
    
    data.train_id = train_id
    data.val_id = val_id
    data.test_id = test_id
    
    if not use_text:
        return data, None
    
    text = {'title': titles, 'abs': abstracts, 'label': labels, 'id': ids}
    # Add in-degree and out-degree to the data object
    data.in_degree = in_degree
    data.out_degree = out_degree
    return data, text
#get_raw_text_arxiv_2023()