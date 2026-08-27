import torch
import pandas as pd
import numpy as np
import torch
import random
import re
from collections import Counter


def get_raw_text_tape_2023(use_text=False, seed=0):

    seed=42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    data = torch.load('dataset/tape_2023/graph.pt')

    # split data
    data.num_nodes = len(data.y)
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    node_id = np.arange(num_nodes)
    np.random.shuffle(node_id)

    train_id = np.sort(node_id[int(num_nodes * 0):int(num_nodes * 0.6)])
    val_id = np.sort(
        node_id[int(num_nodes * 0.6):int(num_nodes * 0.8)])
    test_id = np.sort(node_id[int(num_nodes * 0.8):])

    labels = pd.read_csv('dataset/arxiv_2023_orig/paper_info.csv')['category'].tolist()
    labels = [re.search(r'cs\.\w+', label).group(0) if re.search(r'cs\.\w+', label) else None for label in labels]
    # Define the target labels
    target_labels = ['cs.GT','cs.DC','cs.NI','cs.NE','cs.IR','cs.SI','cs.CY']
    target_labels = ['cs.RO','cs.CL','cs.AI','cs.LG']
    
    # Find the indices of nodes with the target labels
    target_nodes = [i for i, label in enumerate(labels) if label in target_labels]

    # Step 1: Count the number of nodes in each category
    category_counts = Counter([labels[node] for node in target_nodes])

    # Step 2: Determine the smallest category size
    min_category_size = min(category_counts.values())

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
    #other_id = [node for node in target_nodes if node not in train_id + val_id + test_id]
    
    # Filter edge_index to only include edges where both nodes are in target_nodes
    filtered_edge_index = []
    for i in range(edge_index.size(1)):
        if edge_index[0, i].item() in target_nodes_set and edge_index[1, i].item() in target_nodes_set:
            filtered_edge_index.append(edge_index[:, i].unsqueeze(1))

    if filtered_edge_index:
        edge_index = torch.cat(filtered_edge_index, dim=1)
    from torch_geometric.utils import degree
    #num_nodes = len(labels)
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
    test_mask = torch.tensor([x in test_id for x in range(num_nodes)])

    # Remove the sampled nodes from the validation and training sets
    val_id = [node for node in val_id if node not in additional_test_nodes]
    train_id = [node for node in train_id if node not in additional_test_nodes]
    data.train_id = train_id
    data.val_id = val_id
    data.test_id = test_id
    data.train_mask = torch.tensor([x in train_id for x in range(num_nodes)])
    data.val_mask = torch.tensor([x in val_id for x in range(num_nodes)])
    data.test_mask = torch.tensor([x in test_id for x in range(num_nodes)])
    data.in_degree = in_degree
    data.out_degree = out_degree
    data.edge_index = edge_index
    # data.edge_index = data.adj_t.to_symmetric()
    if not use_text:
        return data, None

    df = pd.read_csv('dataset/arxiv_2023_orig/paper_info.csv')
    text = {}
    text['title'] = df['title'].tolist()
    text['abs'] = df['abstract'].tolist()
    text['label'] = df['category'].tolist()
    text['id'] = df['arxiv_id'].tolist()
    cs_labels = [re.search(r'cs\.\w+', label).group(0) if re.search(r'cs\.\w+', label) else None for label in text['label']]
    text['label'] = cs_labels
    return data, text

#get_raw_text_tape_2023(use_text=True)
