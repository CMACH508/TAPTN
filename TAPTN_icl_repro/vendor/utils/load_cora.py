"""
Reference: https://github.com/XiaoxinHe/TAPE/blob/main/core/data_utils/load_cora.py
"""

import os
import sys
sys.path.append('../')

import numpy as np
import torch
import random

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch_geometric.utils as pyg_utils
import torch_geometric
torch.serialization.add_safe_globals([torch_geometric.data.data.Data])

# return cora dataset as pytorch geometric Data object together with 60/20/20 split, and list of cora IDs

cora_mapping = {
    0: "Case Based",
    1: "Genetic Algorithms",
    2: "Neural Networks",
    3: "Probabilistic Methods",
    4: "Reinforcement Learning",
    5: "Rule Learning",
    6: "Theory"
}


def get_cora_casestudy(SEED=0):
    """
    Loads the Cora dataset and performs a 60/20/20 split.

    Args:
        SEED (int): Random seed for reproducibility.

    Returns:
        data (torch_geometric.data.Data): Cora dataset as a PyTorch Geometric Data object.
        data_citeid (list): List of Cora IDs.
    """
    data_X, data_Y, data_citeid, data_edges = parse_cora()

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    data_name = 'cora'
    _root = os.path.join(os.environ['TAPTN_ASSETS'], 'dataset') if os.environ.get('TAPTN_ASSETS') else 'dataset'
    dataset = Planetoid(_root, data_name, transform=T.NormalizeFeatures())
    data = dataset[0]
    

    data.x = torch.tensor(data_X).float()
    data.edge_index = torch.tensor(data_edges).long()
    data.y = torch.tensor(data_Y).long()
    data.num_nodes = len(data_Y)

    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    data.train_id = np.sort(node_id[:int(data.num_nodes * 0.1)])
    data.val_id = np.sort(node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
    data.test_id = np.sort(node_id[int(data.num_nodes * 0.8):])

    data.train_mask = torch.tensor([x in data.train_id for x in range(data.num_nodes)])
    data.val_mask = torch.tensor([x in data.val_id for x in range(data.num_nodes)])
    data.test_mask = torch.tensor([x in data.test_id for x in range(data.num_nodes)])
    # Calculate the in-degree of every node and store it in the data object
    data.in_degree = pyg_utils.degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    data.out_degree = pyg_utils.degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)
    data2 = data.clone()
    data2.edge_index = torch.tensor(np.unique(np.flipud(data_edges),axis=1)).long()

    return data, data2, data_citeid


def parse_cora():
    """
    Parses the Cora dataset files and returns the necessary data.

    Returns:
        data_X (numpy.ndarray): Feature matrix of the Cora dataset.
        data_Y (numpy.ndarray): Label array of the Cora dataset.
        data_citeid (numpy.ndarray): Array of Cora IDs.
        data_edges (numpy.ndarray): Edge array of the Cora dataset.
    """
    path = os.path.join(os.environ.get('TAPTN_ASSETS', ''), 'dataset/cora/cora').rstrip('/') if os.environ.get('TAPTN_ASSETS') else 'dataset/cora/cora'
    idx_features_labels = np.genfromtxt("{}.content".format(path), dtype=np.dtype(str))
    data_X = idx_features_labels[:, 1:-1].astype(np.float32)
    labels = idx_features_labels[:, -1]
    class_map = {x: i for i, x in enumerate(['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
                                            'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory'])}
    data_Y = np.array([class_map[l] for l in labels])
    data_citeid = idx_features_labels[:, 0]
    idx = np.array(data_citeid, dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}.cites".format(path), dtype=np.dtype(str))
    #replacing each item in the edges_unordered array with its corresponding value in the idx_map dictionary, while preserving the original shape of the array.
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
    #remove rows with None
    data_edges = np.array(edges[~(edges == None).max(1)], dtype='int')
    # #undirectedize the graph
    #data_edges = np.vstack((data_edges, np.fliplr(data_edges)))
    # Randomly select a portion of edges
    num_edges = data_edges.shape[0]
    num_to_flip = int(num_edges * 0.4)  # Change this to the portion you want
    np.random.seed(42)
    # indices_to_flip = np.random.choice(num_edges, num_to_flip, replace=False)

    # #Flip the source and destination of the selected edges
    # data_edges[indices_to_flip] = data_edges[indices_to_flip][:, ::-1]
    #data_edges = data_edges[:, ::-1]

    return data_X, data_Y, data_citeid, np.unique(data_edges, axis=0).transpose()


def get_raw_text_cora(use_text=False, seed=0):
    """
    Retrieves the raw text data for the Cora dataset.

    Args:
        use_text (bool): Whether to include the text data or not.
        seed (int): Random seed for reproducibility.

    Returns:
        data (torch_geometric.data.Data): Cora dataset as a PyTorch Geometric Data object.
        text (dict): Dictionary containing the raw text data (title, abstract, label).
    """
    data, data2, data_citeid = get_cora_casestudy(seed)
    if not use_text:
        return data, data2, None

    _mcc = os.path.join(os.environ['TAPTN_ASSETS'], 'dataset/cora/mccallum/cora') if os.environ.get('TAPTN_ASSETS') else 'dataset/cora/mccallum/cora'
    with open(os.path.join(_mcc, 'papers')) as f:
        lines = f.readlines()
    pid_filename = {}
    for line in lines:
        pid = line.split('\t')[0]
        fn = line.split('\t')[1].replace(':', '_')
        pid_filename[pid] = fn

    path = os.path.join(_mcc, 'extractions') + os.sep

    text = {'title': [], 'abs': [], 'label': []}

    all_files = {f.lower(): f for f in os.listdir(path)}

    for pid in data_citeid:
        expected_fn = pid_filename[pid].lower()
        if expected_fn in all_files:
            real_fn = all_files[expected_fn]
            with open(path+real_fn) as f:
                lines = f.read().splitlines()

            for line in lines:
                if 'Title:' in line:
                    ti = line
                if 'Abstract:' in line:
                    ab = line
            text['title'].append(ti)
            text['abs'].append(ab)

    for i in range(len(data.y)):
        text['label'].append(cora_mapping[data.y[i].item()])

    # import json
    # for i in range(len(text['abs'])):
    #     text['abs'][i]=json.load(open(f'dataset/cora/gpt_responses/cora_nosem/{i}.json','r'))['choices'][0]['message']['content']
    return data, data2, text
