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

def load_wisconsin(data_set_name):
    _assets = os.environ.get("TAPTN_ASSETS", "")
    if not _assets:
        raise RuntimeError(
            "TAPTN_ASSETS is not set. Unpack TAPTN_rewiring_repro_assets and run "
            "export TAPTN_ASSETS=/path/to/TAPTN_rewiring_repro_assets (see README)."
        )
    extracted_path = os.path.join(_assets, "webkb-data")
    if not os.path.isdir(extracted_path):
        raise FileNotFoundError(f"WebKB dump not found at {extracted_path}")

    print("Looking for HTML files in Wisconsin directories...")

    html_files = []
    for root, dirs, files in os.walk(extracted_path):
        if data_set_name in dirs:
            target_path = os.path.join(root, data_set_name)
            for w_root, _, w_files in os.walk(target_path):
                for file in w_files:
                    html_files.append(os.path.join(w_root, file))

    print(f"Found {len(html_files)} HTML files.")

    _order_candidates = [
        os.path.join(_assets, f"webkb_html_order_{data_set_name}.txt"),
        os.path.join(_assets, "pkls", f"webkb_html_order_{data_set_name}.txt"),
        f"webkb_html_order_{data_set_name}.txt",
    ]
    _order_path = next((p for p in _order_candidates if os.path.isfile(p)), None)
    if _order_path:
        wanted = [ln.strip() for ln in open(_order_path) if ln.strip()]
        by_rel = {}
        for p in html_files:
            rel = p
            marker = f"{os.sep}webkb-data{os.sep}"
            if marker in p:
                rel = p.split(marker, 1)[1]
            by_rel[rel.replace("\\", "/")] = p
        ordered = [by_rel[rel.replace("\\", "/")] for rel in wanted if rel.replace("\\", "/") in by_rel]
        if len(ordered) == len(html_files):
            html_files = ordered
        else:
            print(f"[load_wisconsin] html order file incomplete "
                  f"(matched={len(ordered)} n={len(html_files)}); using walk order")

    # Step 3: Generate the wordbag/tf-idf encoding for each webpage
    documents = []
    file_indices = {}
    index = 0
    test_mask=[]
    for html_file in html_files:
        with open(html_file, 'r', encoding='latin1') as file:
            content = file.read()
            # Remove HTML tags
            text = re.sub('<[^<]+?>', '', content)
            documents.append(text)
            url_path = html_file.split('/')[-1].replace('^','/')
            url_path = url_path.rstrip('/')  # Remove trailing '/' if present
            file_indices[url_path] = index
            if html_file.split('/')[-3]!="other":
                test_mask.append(True)
            else:
                test_mask.append(False)
            index += 1

    print("Generating TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(documents)
    X = torch.tensor(X.toarray(), dtype=torch.float)
    print(f"TF-IDF feature matrix shape: {X.shape}")

    # Assuming 'vectorizer' and 'X' are already defined as shown in the excerpt
    # 兼容 scikit-learn 新旧版本
    if hasattr(vectorizer, 'get_feature_names_out'):
        feature_names = vectorizer.get_feature_names_out()
    else:
        feature_names = vectorizer.get_feature_names()
    
    text = []  # Initialize the 'text' dictionary
    abstract = []  # Initialize the 'abstract' dictionary
    labels = []  # Initialize the 'labels' dictionary
    urls=[] # Initialize the 'urls' dictionary
    descriptions = []  # Initialize the 'descriptions' dictionary
    # Convert TF-IDF sparse matrix to dense for easier manipulation
    X_dense = X.numpy()
    _abs_candidates = [
        os.path.join(_assets, f"file_abs_{data_set_name}.pkl"),
        f"file_abs_{data_set_name}.pkl",
    ]
    _abs_path = next((p for p in _abs_candidates if os.path.isfile(p)), None)
    if _abs_path is None:
        raise FileNotFoundError(f"file_abs_{data_set_name}.pkl not found under TAPTN_ASSETS or cwd")
    file_abs = pickle.load(open(_abs_path, "rb"))

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
    out_degree = degree(edge_index[1], num_nodes=X.size(0), dtype=torch.long)
    in_degree = degree(edge_index[0], num_nodes=X.size(0), dtype=torch.long)

    # Step 5: Construct the PyG Data object
    data = Data(x=X, edge_index=edge_index,test_mask=torch.tensor(test_mask),test_id=torch.tensor([i for i in range(len(test_mask)) if test_mask[i]]),train_mask=torch.tensor([False for _ in test_mask]),train_id=torch.tensor([]), val_mask=torch.tensor([False for _ in test_mask]),val_id=torch.tensor([]), out_degree=out_degree, in_degree=in_degree)
    

    return data, text

# # Save the PyG data object
# output_path = '/mnt/data/wisconsin_data.pt'
# torch.save(data, output_path)
# print(f"Data object saved to {output_path}")

# #output_path
