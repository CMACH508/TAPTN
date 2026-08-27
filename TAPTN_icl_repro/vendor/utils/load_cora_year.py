"""
Reference: https://github.com/XiaoxinHe/TAPE/blob/main/core/data_utils/load_cora.py
"""

import os
import sys
import re
sys.path.append('../')

import numpy as np
import torch
import random

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch_geometric.utils as pyg_utils

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
    dataset = Planetoid('dataset', data_name, transform=T.NormalizeFeatures())
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
    path = 'dataset/cora/cora'
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
    data_edges = data_edges[:, ::-1]

    return data_X, data_Y, data_citeid, np.unique(data_edges, axis=0).transpose()

def remove_outliers_and_partition(data):
    # Step 1: Remove outliers
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    non_outliers = [x for x in data if lower_bound <= x <= upper_bound]
    
    # Step 2: Sort the non-outlier elements
    sorted_non_outliers = sorted(non_outliers)
    
    # Step 3: Divide into five equal parts
    part_size = len(sorted_non_outliers) // 5
    partitions = [sorted_non_outliers[i:i + part_size] for i in range(0, len(sorted_non_outliers), part_size)]
    
    # Adjust the last partition to include any remaining elements
    if len(partitions) > 5:
        partitions[4].extend(partitions.pop())
    
    return [partitions[i][0] for i in range(5)]+[partitions[-1][-1]]

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

    with open('dataset/cora/mccallum/cora/papers')as f:
        lines = f.readlines()
    pid_filename = {}
    for line in lines:
        pid = line.split('\t')[0]
        fn = line.split('\t')[1].replace(':', '_')
        pid_filename[pid] = fn

    path = 'dataset/cora/mccallum/cora/extractions/'

    text = {'title': [], 'abs': [], 'label': []}

    all_files = {f.lower(): f for f in os.listdir(path)}

    # Initialize a dictionary to hold pid to year mapping
    pid_year = {}
    year_from_extra = {'1131274':1999,'682666':1998,'1123188':1995,'399173':1996,'782486':1997,'1153891':1995,'51834':1995,'200630':1996,"1126029":1996,'198866':1994,'1118017':1994,"1113739":1999,'242637':1997,'1152272':1997,'1153703':1991,'1112319':2000,'1153786':1991,'1105062':1991,'1154500':1997,'1131277':1998,'1122574':1997,'1152307':1995, '1136310':1997,'1105116':1998,'124064':1994,'1112426':1999,'1131314':1998,'1129835':1998,'1132083':2000,'132806':1991, '1107312':1997,'1108050':1993,'1113934':1996,'1132809':1995,'1153148':1991,'1153166':1995,'1136422':1994,'1124844':1999,'1153195':1993,'1119751':2003,'1110390':1998,'1115456':1994,'1107455':2000,'1132922':1997,'1117653':1996,'1132968':1996,'1128542':1995, '52000':1993, '52003':1993, '52007':1994, '1131550':1997, '1112686':1997,'1123553':2000,'1135137':1996, '1130856':2003, '1123576':1996, '1103979':1996, '1119140':1997,'75318':1970,'1106112':1997,'1131639':1995,'20942':1997,'390896':1999,'1154103':1991,'1104787':1995,'1120019':1997,'1129443':1998,'1104055':1997,'1113541':1997,'1104809':2000,'1132459':1995,'1131745':1996,'1154230':1993,'1152821':1995,'1128839':1996,'1128846':1996,'1140547':2001,'1152162':1996,'646809':1995,'1152194':2007,'1115886':1994,'1152904':1996,'289780':1999, '1152975':1998, '1104261':1994, '1131165':1997,'1107171':1997,'1128204':1992,'1128208':1998,'1127530':1997,'1128267':1999,'1127551':1997,'1128291':1998,'1103676':1997,'1132706':1995,'1133469':1998,'1125467':1998,'1109439':1996, '1128319':1998,'1152379':1999,'1136342':1996,'1131300':1999,'1131330':1998,'1103737':1999,'1130637':1987,'1131374':1997,'1130653':1996,'1130657':1996,'1114629':1996,'1117501':1997,'1153160':1991,'1109542':1999,'1125597':1997,'1128430':1995,'1103031':2001,'1121254':1997,'1131466':1998,'1118347':1997,'1153262':1985,'19231':1992,'1132948':1996,'1121313':1998,'1130080':1997,'1103969':1994,'1107572':1994,'1154076':1995,'1152663':1997,'1110515':1996,'1129368':1998,'1110579':1999,'1133028':1999,'1133047':1995,'1105433':1996,'1130927':1999,'1130929':1992,'1130931':1999,'1152714':1996,'1152740':1999,'1117833':1995,'1132434':1996,'1131734':1997,'1152075':1997,'1153577':1993,'1125992':1997,'1114331':1997,'1113614':1997,'1114364':1995,'1105672':1997,'1152944':1999,'1118764':1997,'1121739':1996,'1126037':1997,'1126044':1998,'1105718':1997,'1105764':1996,'1109392':1999,'1115959':1997,'20528':1997,'1129778':1998,'1114512':1997,'1114526':1998,'1153064':2000,'1134197':1999,'1128314':1999,'1152394':1995,'1154520':1995,'1129096':1997,'1110256':1997,'1153853':1997,'1153899':1997,'1131335':1994,'1107319':1995,'1129111':1990,'576362':1996,'1128453':2000,'1109581':1998,'1153900':1998,'1153942':1996,'1131471':1997,'1107418':1996, '1134320':1996, '1108169':1999,'1126315':1990,'1108175':1995,'1118302':1995,'1130780':1999,'1134348':1998, '240791':1993,'1102400':1996,'1127863':1996,'1120650':1997,'1105360':1997,'1123530':1998,'1131557':1995,'1131565':1995,'1138027':1995,'1154068':1998,'1152633':1992,'1117786':1997,'1110520':1996,'1111265':1997,'1112767':1997,'180301':1995,'390889':1998,'1115670':1997,'1114992':1999,'1119295':1997,'1102625':1997,'1103383':1996,'1113551':1996,'1132461':1995,'1131741':1996,'1131752':1995,'1107728':1990,'1154232':1990,'1154233':1994,'1115790':1997,'1125953':1995,'1120197':1993,'1114336':1994,'1140548':1995,'1139009':1999,'1129610':1998,'1131150':1999,'1131163':1998,'1131172':1997,'1131180':2000,'1128977':1998,'59626':1993,'1131270':1999,'1131420':1995,'1130808':1996,'1130934':1997,'1131828':1997,'1106401':2000,'1107136':1997, '1131257':1998,'1106568':1998,'1107325':1998,'1105231':1995,'286500':1996,'1120786':1995,'1110028':1995,'1127812':1996,'1107567':1995,'1120777':1997,'1128997':1998,'1134056':1998,'1131258':1998,'1114502':1996,'1131348':1993,'1130678':1998,'531348':1997,'1129208':1995,'1154042':1997,'1134022':1995,'1119505':1998,'19045':1996,'9559':1995,'1135750':1997,'529165':1997,'1127810':1997,'1121459':1997,'267003':1996}
    # Regular expression to find the year within the <year> tags
    year_regex = re.compile(r'<year>.*?(\d{4}).*?</year>')

    with open('dataset/cora/mccallum/cora/papers') as f:
        for line in lines:
            parts = line.split('\t')
            if parts[2] == '\n':
                continue
            pid = parts[0]
            line = '\t'.join(parts[1:])
            # Search for the year using the regular expression
            year_match = year_regex.findall(line)
            if year_match:
                for year in year_match:
                    if 1912 < int(year) < 2000:
                        pid_year[pid] = int(year)
                        break  # Break the loop once a suitable year is found
                    else:
                        pass
            if pid not in pid_year:
                year_regex2=re.compile(r'\b19(\d{2})\D*')
                year_match = year_regex2.findall(line)
                for year in year_match:
                    if 60 <= int(year) <= 99:
                        pid_year[pid] = 1900+int(year)
                        break  # Break the loop once a suitable year is found
                    else:
                        pass

    # Convert years to a NumPy array
    years = np.array(list(map(int, pid_year.values())))

    # Determine the range and calculate interval size
    min_year, max_year = years.min(), years.max()
    interval_size = (max_year - min_year) / 5

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
    #         if len(text['title'])-1 in [56,  190,  255,  305,  385,  390,  590,  694,  716, 1108, 1739,
    #    1918, 2077, 2699]:
    #             pass
            if pid in pid_year:
                year = pid_year[pid]
                label = np.floor((pid_year[pid] - min_year) / interval_size).astype(int)
            else:
                year = -1
                year_regex = re.compile(r'19(\d{2})(?!\d)|(?<![:\[\d])(\d{2})(?!\d)')
                year_match = year_regex.findall(ti)
                if year_match:
                    year_match = [element for tupl in year_match for element in tupl if element]
                    for year1 in year_match:
                        if 60 <= int(year1) <= 99:
                            year = 1900 + int(year1)
                            break
                if year != -1:
                    label = np.floor((year - min_year) / interval_size).astype(int)
                else:
                    for line in lines:
                        if 'URL' in line or 'Date:' in line or 'Note:' in line:
                            year_match = year_regex.findall(line)
                            if year_match:
                                year_match = [element for tupl in year_match for element in tupl if element]
                                for year1 in year_match:
                                    if 60 <= int(year1) <= 99:
                                        year = 1900 + int(year1)
                                        break   
                if year == -1:
                    if pid in year_from_extra:
                        year = year_from_extra[pid]
                    else:
                        continue
                label = np.floor((year - min_year) / interval_size).astype(int)
            if label == 5:
                label = 4
            if pid in year_from_extra:
                year=year_from_extra[pid]
            text['label'].append(year)

    partition_points=remove_outliers_and_partition(np.array(text['label']).astype(int)[data.test_mask.numpy()])
    transformed_labels = []
    ps = partition_points[0]  # Smallest partition point
    pb = partition_points[-1]  # Biggest partition point
    
    for label in np.array(text['label']).astype(int):
        if label < ps:
            transformed_labels.append(f'earlier than {ps}')
        elif label >= pb:
            transformed_labels.append(f'later than {pb}')
        else:
            for i in range(len(partition_points) - 1):
                if partition_points[i] <= label < partition_points[i + 1]:
                    transformed_labels.append(f'{partition_points[i]}-{partition_points[i + 1]-1}')
                    break
    text['label'] = transformed_labels
    # for i in range(len(data.y)):
    #     text['label'].append(cora_mapping[data.y[i].item()])



    return data, data2, text
