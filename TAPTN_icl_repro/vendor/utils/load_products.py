"""
mapping references:
https://github.com/CurryTang/Graph-LLM/blob/master/utils.py
"""

from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch
import functools
import pandas as pd
from datasets import load_dataset
import torch_geometric

# PyTorch >= 2.4 将 weights_only=True 设为默认值，但 OGB/PyG 保存的数据文件
# 包含多种自定义类，无法仅靠 add_safe_globals 完整枚举。
# 此处在模块级别临时覆盖 torch.load，使其默认 weights_only=False，
# 以确保 PygNodePropPredDataset 及其他 PyG 数据加载正常工作。
_orig_torch_load = torch.load
torch.load = functools.partial(_orig_torch_load, weights_only=False)


def get_raw_dataset(raw_train="dataset/ogbn_products/Amazon-3M.raw/trn.json", 
                    raw_test="dataset/ogbn_products/Amazon-3M.raw/tst.json",
                    label2cat="dataset/ogbn_products/mapping/labelidx2productcategory.csv",
                    idx2asin="dataset/ogbn_products/mapping/nodeidx2asin.csv"):
    
    train_part = load_dataset("json", data_files=raw_train)
    test_part = load_dataset("json", data_files=raw_test)
    train_df = train_part['train'].to_pandas()
    test_df = test_part['train'].to_pandas()
    combine_df = pd.concat([train_df, test_df], ignore_index=True)
    
    label2cat_df = pd.read_csv(label2cat)
    idx2asin_df = pd.read_csv(idx2asin)
    
    
    idx_mapping = {row[0]: row[1] for row in idx2asin_df.values}
    label_mapping = {row['label idx']: row['product category'] for _, row in label2cat_df.iterrows()}
    content_mapping = {row[0]: (row[1], row[2]) for row in combine_df.values}
    
    return idx_mapping, content_mapping, label_mapping

def get_raw_text_products(use_text=False, seed=0):
    dataset = PygNodePropPredDataset(name='ogbn-products')
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

    from torch_geometric.utils import degree
    num_nodes = data.num_nodes
    data.out_degree = degree(data.edge_index[0], num_nodes=num_nodes, dtype=torch.long)
    data.in_degree = degree(data.edge_index[1], num_nodes=num_nodes, dtype=torch.long)

    from torch_geometric.utils import subgraph, to_undirected, to_networkx
    import networkx as nx
    #from scipy.sparse import csr_matrix
    #from scipy.sparse.csgraph import connected_components
    #import numpy as np
    # Find the largest connected component
    #edge_index = data.edge_index.numpy()
    num_nodes = data.num_nodes
    torch.manual_seed(42)
    subset = torch.randperm(num_nodes)[:50000]
    edge_index = subgraph(subset, data.edge_index, relabel_nodes=False)[0]
     # Make the graph undirected
    data.edge_index = to_undirected(edge_index)
    #import torch
    import torch_geometric.transforms as T
    import utils.LCC as LCC
     # Apply the LargestConnectedComponents transform
    transform = LCC.LargestConnectedComponents()
    data = transform(data)


    if not use_text:
        return data, None

    idx_mapping, content_mapping, label_mapping = get_raw_dataset()

    text = {'title': [], 'content': [], 'label': []}

    for i in range(len(data.y)):
        uid = idx_mapping.get(i, None)
        if uid:
            title, content = content_mapping.get(uid, (None, None))
            label = label_mapping.get(data.y[i].item(), None)
            
            text['title'].append(title)
            text['content'].append(content)

            mapped_label = products_mapping.get(label, None)
            # assert mapped_label is not None, f"Label {label} not found in mapping"
            if mapped_label is None:
                text['label'].append('label 25')
            else:
                text['label'].append(mapped_label)

    
    return data, text


products_mapping = {'Home & Kitchen': 'Home & Kitchen',
        'Health & Personal Care': 'Health & Personal Care',
        'Beauty': 'Beauty',
        'Sports & Outdoors': 'Sports & Outdoors',
        'Books': 'Books',
        'Patio, Lawn & Garden': 'Patio, Lawn & Garden',
        'Toys & Games': 'Toys & Games',
        'CDs & Vinyl': 'CDs & Vinyl',
        'Cell Phones & Accessories': 'Cell Phones & Accessories',
        'Grocery & Gourmet Food': 'Grocery & Gourmet Food',
        'Arts, Crafts & Sewing': 'Arts, Crafts & Sewing',
        'Clothing, Shoes & Jewelry': 'Clothing, Shoes & Jewelry',
        'Electronics': 'Electronics',
        'Movies & TV': 'Movies & TV',
        'Software': 'Software',
        'Video Games': 'Video Games',
        'Automotive': 'Automotive',
        'Pet Supplies': 'Pet Supplies',
        'Office Products': 'Office Products',
        'Industrial & Scientific': 'Industrial & Scientific',
        'Musical Instruments': 'Musical Instruments',
        'Tools & Home Improvement': 'Tools & Home Improvement',
        'Magazine Subscriptions': 'Magazine Subscriptions',
        'Baby Products': 'Baby Products',
        'label 25': 'label 25',
        'Appliances': 'Appliances',
        'Kitchen & Dining': 'Kitchen & Dining',
        'Collectibles & Fine Art': 'Collectibles & Fine Art',
        'All Beauty': 'All Beauty',
        'Luxury Beauty': 'Luxury Beauty',
        'Amazon Fashion': 'Amazon Fashion',
        'Computers': 'Computers',
        'All Electronics': 'All Electronics',
        'Purchase Circles': 'Purchase Circles',
        'MP3 Players & Accessories': 'MP3 Players & Accessories',
        'Gift Cards': 'Gift Cards',
        'Office & School Supplies': 'Office & School Supplies',
        'Home Improvement': 'Home Improvement',
        'Camera & Photo': 'Camera & Photo',
        'GPS & Navigation': 'GPS & Navigation',
        'Digital Music': 'Digital Music',
        'Car Electronics': 'Car Electronics',
        'Baby': 'Baby',
        'Kindle Store': 'Kindle Store',
        'Buy a Kindle': 'Buy a Kindle',
        'Furniture & D&#233;cor': 'Furniture & Decor',
        '#508510': '#508510'}

products_keys_list = list(products_mapping.keys())
