import random
import tarfile
import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree
import torch_geometric
import pickle

def load_wisconsin(seed, data_set_name, use_text=True):
    #data_set_name = 'texas'
    # Step 1: Extract the data from the provided `webkb-data.gtar.gz` file
    _assets = os.environ.get('TAPTN_ASSETS', '')
    if not _assets:
        raise RuntimeError(
            'TAPTN_ASSETS is not set. Unpack the asset bundle and run '
            'export TAPTN_ASSETS=/path/to/TAPTN_finetune_repro_assets (see README).')
    tar_path = os.path.join(_assets, 'webkb-data.gtar.gz')
    extracted_path = os.path.join(_assets, 'webkb-data')

    # Extract the tar file
    print("Extracting tar file...")
    # with tarfile.open(tar_path, 'r:gz') as tar:
    #     tar.extractall(path=extracted_path)

    # Step 2: Parse the extracted data to identify Wisconsin university webpages
    print("Looking for HTML files in Wisconsin directories...")

    html_files = []
    # Step 1: Walk through the directory structure
    for root, dirs, files in os.walk(extracted_path):
        # Step 2: Check if 'wisconsin' is one of the subdirectories
        if data_set_name in dirs:
            target_path = os.path.join(root, data_set_name)
            # Step 3: If a 'wisconsin' directory is found, list all HTML files within it
            for w_root, _, w_files in os.walk(target_path):
                for file in w_files:
                    html_files.append(os.path.join(w_root, file))

    print(f"Found {len(html_files)} HTML files.")

    # Reorder to the walk order used when LM .pred files were written, if recorded.
    _order_candidates = []
    if _assets:
        _order_candidates += [
            os.path.join(_assets, f'webkb_html_order_{data_set_name}.txt'),
            os.path.join(_assets, 'pkls', f'webkb_html_order_{data_set_name}.txt'),
        ]
    _order_candidates.append(f'webkb_html_order_{data_set_name}.txt')
    _order_path = next((p for p in _order_candidates if os.path.isfile(p)), None)
    if _order_path:
        wanted = [ln.strip() for ln in open(_order_path) if ln.strip()]
        # html_files currently absolute; match by suffix relative to webkb-data
        by_rel = {}
        for p in html_files:
            rel = p
            marker = f'{os.sep}webkb-data{os.sep}'
            if marker in p:
                rel = p.split(marker, 1)[1]
            by_rel[rel.replace('\\', '/')] = p
        ordered = []
        missing = 0
        for rel in wanted:
            rel_n = rel.replace('\\', '/')
            if rel_n in by_rel:
                ordered.append(by_rel[rel_n])
            else:
                missing += 1
        if missing == 0 and len(ordered) == len(html_files):
            html_files = ordered
        else:
            print(f'[load_wisconsin] html order file incomplete '
                  f'(matched={len(ordered)} missing={missing} n={len(html_files)}); using walk order')

    # Step 3: Generate the wordbag/tf-idf encoding for each webpage
    documents = []
    file_indices = {}
    index = 0
    test_mask=[]
    for html_file in html_files:
        url_path = html_file.split('/')[-1].replace('^','/')
        url_path = url_path.rstrip('/')  # Remove trailing '/' if present
        file_indices[url_path] = index
        if use_text:
            with open(html_file, 'r', encoding='latin1') as file:
                content = file.read()
            text = re.sub('<[^<]+?>', '', content)
            documents.append(text)
        else:
            documents.append('')
        if html_file.split('/')[-3]!="other":
            test_mask.append(True)
        else:
            test_mask.append(False)
        index += 1
    test_id = np.array([i for i in range(len(test_mask)) if test_mask[i]])
    random.seed(seed)
    np.random.seed(seed)
    np.random.shuffle(test_id)
    train_id, val_id, test_id = test_id[:int(len(test_id)*0.6)], test_id[int(len(test_id)*0.6):int(len(test_id)*0.8)], test_id[int(len(test_id)*0.8):]
    test_mask=[]
    train_mask=[]
    val_mask=[]
    for i in range(len(html_files)):
        if i in train_id:
            train_mask.append(True)
            val_mask.append(False)
            test_mask.append(False)
        elif i in val_id:
            train_mask.append(False)
            val_mask.append(True)
            test_mask.append(False)
        elif i in test_id:
            train_mask.append(False)
            val_mask.append(False)
            test_mask.append(True)
        else:
            train_mask.append(False)
            val_mask.append(False)
            test_mask.append(False)
    # Fast path: dry-run / GNN-only needs labels, splits and edges, not TF-IDF text.
    if not use_text:
        class_map = {x: i for i, x in enumerate(
            ['project', 'course', 'faculty', 'student', 'staff', 'department', 'other'])}
        labels = [html_files[i].split('/')[-3] for i in range(len(html_files))]
        data_Y = np.array([class_map[l] for l in labels])
        file_indices = {}
        for i, html_file in enumerate(html_files):
            url_path = html_file.split('/')[-1].replace('^', '/').rstrip('/')
            file_indices[url_path] = i
        edges = []
        for html_file in html_files:
            with open(html_file, 'r', encoding='latin1') as file:
                content = file.read()
            links = re.findall(r'href=\s*[\'"]?([^\'" >]+)(?:[\'"])?', content, re.IGNORECASE)
            src_url = html_file.split('/')[-1].replace('^', '/').rstrip('/')
            src = file_indices[src_url]
            for link in links:
                if link.rstrip('/') in file_indices:
                    edges.append((src, file_indices[link.rstrip('/')]))
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_index = edge_index[[1, 0], :]
        edge_index, _ = torch_geometric.utils.coalesce(edge_index, None, num_nodes=len(html_files))
        edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)
        data = Data(
            x=torch.zeros((len(html_files), 1)),
            edge_index=edge_index,
            test_mask=torch.tensor(test_mask),
            train_mask=torch.tensor(train_mask),
            val_mask=torch.tensor(val_mask),
            y=torch.tensor(data_Y).long(),
        )
        return data, None

    print("Generating TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(documents)
    X = torch.tensor(X.toarray(), dtype=torch.float)
    print(f"TF-IDF feature matrix shape: {X.shape}")

    # Assuming 'vectorizer' and 'X' are already defined as shown in the excerpt
    feature_names = (
        vectorizer.get_feature_names_out()
        if hasattr(vectorizer, "get_feature_names_out")
        else vectorizer.get_feature_names()
    )
    text = []  # Initialize the 'text' dictionary
    abstract = []  # Initialize the 'abstract' dictionary
    labels = []  # Initialize the 'labels' dictionary
    urls=[] # Initialize the 'urls' dictionary
    descriptions = []  # Initialize the 'descriptions' dictionary
    # Convert TF-IDF sparse matrix to dense for easier manipulation
    X_dense = X.numpy()
    _fab_candidates = [
        f'file_abs_{data_set_name}.pkl',
        os.path.join(_assets, f'file_abs_{data_set_name}.pkl') if _assets else '',
        os.path.join(_assets, 'pkls', f'file_abs_{data_set_name}.pkl') if _assets else '',
    ]
    _fab_path = next((p for p in _fab_candidates if p and os.path.isfile(p)), f'file_abs_{data_set_name}.pkl')
    file_abs = pickle.load(open(_fab_path, 'rb'))

    for i, doc in enumerate(documents):
        # Get the indices of the top 100 words based on TF-IDF scores
        top_indices = np.argsort(X_dense[i])[-100:][::-1]
        top_words = feature_names[top_indices]
        top_scores = X_dense[i][top_indices]
        
        # Create a description of the top 100 words and their scores
        description = ", ".join([f"{word}: {score:.3f}" for word, score in zip(top_words, top_scores) if score >= 0.001])
        
        # Use the URL or filename as the key
        urls.append(html_files[i].split('/')[-1].replace('^','/').rstrip('/'))
        text.append(description)
        abstract.append(file_abs[html_files[i].split('/')[-1].replace('^','/').rstrip('/')])
        descriptions.append(description)
        labels.append(html_files[i].split('/')[-3])
    text = {'title':urls, 'abs':abstract, 'dsc':descriptions, 'label':labels}
    class_map = {x: i for i, x in enumerate(['project', 'course', 'faculty', 'student', 'staff','department','other'])}
    data_Y = np.array([class_map[l] for l in labels])
    text2 = []
    for i in range(len(text['title'])):
        text2.append(f"{text['title'][i]}\n{text['abs'][i]}")

    print("Text descriptions generated for each webpage.")

    # Step 4: Create edges representing hyperlinks between webpages
    edges = []
    for html_file in html_files:
        with open(html_file, 'r', encoding='latin1') as file:
            content = file.read()
            links = re.findall(r'href=\s*[\'"]?([^\'" >]+)(?:[\'"])?', content, re.IGNORECASE)
            src_url = html_file.split('/')[-1].replace('^','/').rstrip('/')
            src = file_indices[src_url]
            for link in links:           
                #full_link = os.path.join(os.path.dirname(html_file), link)
                if link.rstrip('/') in file_indices:
                    dst = file_indices[link.rstrip('/')]
                    edges.append((src, dst))

    print(f"Found {len(edges)} edges.")

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = edge_index[[1,0],:]
    
    # Coalesce the edges to remove duplicates and sort them
    edge_index, _ = torch_geometric.utils.coalesce(edge_index, None, num_nodes=X.size(0))

    # Remove self-loops from the edge_index
    edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)

    # Compute in-degree and out-degree for each node
    out_degree = degree(edge_index[0], num_nodes=X.size(0), dtype=torch.long)
    in_degree = degree(edge_index[1], num_nodes=X.size(0), dtype=torch.long)

    # Step 5: Construct the PyG Data object
    data = Data(x=X, edge_index=edge_index,test_mask=torch.tensor(test_mask),test_id=torch.tensor(test_id),train_mask=torch.tensor(train_mask),train_id=torch.tensor(train_id), val_mask=torch.tensor(val_mask),val_id=torch.tensor(val_id), out_degree=out_degree, in_degree=in_degree, y= torch.tensor(data_Y).long())
    

    return data, text2

# # Save the PyG data object
# output_path = '/mnt/data/wisconsin_data.pt'
# torch.save(data, output_path)
# print(f"Data object saved to {output_path}")

# #output_path
