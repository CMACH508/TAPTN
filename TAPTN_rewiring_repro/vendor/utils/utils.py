import random
import re
import torch
import numpy as np
import json
import os
import openai
try:
    from utils.load_cora import get_raw_text_cora
except ImportError:
    def get_raw_text_cora(*args, **kwargs):
        raise RuntimeError("Cora loader is not included in this rewiring package")
try:
    from utils.load_arxiv_2023 import get_raw_text_arxiv_2023
except ImportError:
    def get_raw_text_arxiv_2023(*args, **kwargs):
        raise RuntimeError("arXiv-2023 loader is not included in this rewiring package")
from utils.load_wisconsin import load_wisconsin
from time import sleep
from utils.prompts import generate_system_prompt, arxiv_natural_lang_mapping, refining3_2, refining3_3, refining_actor_iter2, refining5_4, refining5_3, refining5_5, refining_actor
from time import sleep
from random import randint
import threading
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.Agent import Agent

# 创建全局 Agent 实例用于 LLM 交互
# 注意：Agent 类内部已配置 API 密钥和 base_url，无需外部传入
global_agent = Agent(name="TAPTN Assistant", role="Graph Node Classification Expert")

#neighbor_num = pickle.load(open(f"neighbor_num.pkl", "rb"))
def load_data(dataset, use_text=False, seed=0):
    """
    Load data based on the dataset name.

    Parameters:
        dataset (str): Name of the dataset to be loaded. Options are "cora", "pubmed", "arxiv", "arxiv_2023", "actor", and "product".
        use_text (bool, optional): Whether to use text data. Default is False.
        seed (int, optional): Random seed for data loading. Default is 0.

    Returns:
        Tuple: Loaded data and text information.

    Raises:
        ValueError: If the dataset name is not recognized.
    """
    data2 = None
    if dataset == "cora":
        data, data2, text = get_raw_text_cora(use_text, seed)
    elif dataset in ["texas","wisconsin","washington","cornell"]:
        data, text = load_wisconsin(dataset)
    elif dataset == "arxiv_2023":
        data, text = get_raw_text_arxiv_2023(use_text)
    elif dataset == "actor":
        from utils.load_actor import get_raw_text_actor
        data, data2, text = get_raw_text_actor(use_text, seed)
    else:
        raise ValueError("Dataset must be one of: cora, webkb(texas, wisconsin, washington, cornell), arxiv_2023, actor")
    return data, data2, text

def get_subgraph2(node_idx, edge_index, hop=1):
    """
    Get subgraph around a specific node up to a certain hop in a directed graph.

    Parameters:
        node_idx (int): Index of the node.
        edge_index (torch.Tensor): Edge index tensor.
        hop (int, optional): Number of hops around the node to consider. Default is 1.

    Returns:
        list: Lists of nodes for each hop distance, considering the direction of the edges.
    """

    current_nodes = torch.tensor([node_idx])
    all_hops = []

    #process hop 1, mask_src: edge from current node to other nodes, mask_dst: edge from other nodes to current node
    mask_src = torch.isin(edge_index[0], current_nodes)
    mask_dst = torch.isin(edge_index[1], current_nodes)

    # Add both the source and target nodes involved in the edges, new_nodes_src: nodes that current node points to, new_nodes_dst: nodes that point to the current node
    new_nodes_src = torch.unique(edge_index[1][mask_src])
    new_nodes_dst = torch.unique(edge_index[0][mask_dst])

    # Remove the center node from the new nodes
    new_nodes_src = new_nodes_src[new_nodes_src != node_idx]
    new_nodes_dst = new_nodes_dst[new_nodes_dst != node_idx]

    # Store the nodes separately based on the direction of the edge
    all_hops.append(new_nodes_src.tolist())
    all_hops.append(new_nodes_dst.tolist())

    # Update the current nodes for the next iteration
    current_nodes_src = torch.unique(new_nodes_src)
    current_nodes_dst = torch.unique(new_nodes_dst)

    #process hop 2
    if hop == 2:
        #mask_src_src: edge from current node to other nodes, mask_src_dst: edge from other nodes to current node
        mask_src_src = torch.isin(edge_index[0], current_nodes_src)
        mask_src_dst = torch.isin(edge_index[1], current_nodes_src)
        #mask_dst_src: edge from current node to other nodes, mask_dst_dst: edge from other nodes to current node
        mask_dst_src = torch.isin(edge_index[0], current_nodes_dst)
        mask_dst_dst = torch.isin(edge_index[1], current_nodes_dst)

        # Add both the source and target nodes involved in the edges, new_nodes_src_src: nodes that current node src points to, new_nodes_src_dst: nodes that point to the current node src
        new_nodes_src_src = torch.unique(edge_index[1][mask_src_src])
        new_nodes_src_dst = torch.unique(edge_index[0][mask_src_dst])
        new_nodes_dst_src = torch.unique(edge_index[1][mask_dst_src])
        new_nodes_dst_dst = torch.unique(edge_index[0][mask_dst_dst])

        # Remove the center node from the new nodes
        new_nodes_src_src = new_nodes_src_src[new_nodes_src_src != node_idx]
        new_nodes_src_dst = new_nodes_src_dst[new_nodes_src_dst != node_idx]
        new_nodes_dst_src = new_nodes_dst_src[new_nodes_dst_src != node_idx]
        new_nodes_dst_dst = new_nodes_dst_dst[new_nodes_dst_dst != node_idx]

        # Store the nodes separately based on the direction of the edge, new_nodes_src_src: c->c->c, new_nodes_src_dst: c->c<-c, new_nodes_dst_src: c<-c->c, new_nodes_dst_dst: c<-c<-c
        all_hops.append(new_nodes_src_src.tolist())
        all_hops.append(new_nodes_src_dst.tolist())
        all_hops.append(new_nodes_dst_src.tolist())
        all_hops.append(new_nodes_dst_dst.tolist())

    return all_hops

def sample_test_nodes(data, text, sample_size, dataset, node_set='test'):
    """
    Randomly sample nodes for evaluation from specified node set.

    Parameters:
        data: Graph data object.
        text: Textual information associated with nodes.
        sample_size (int): Number of nodes to sample. If None or larger than available nodes, use all.
        dataset (str): Name of the dataset being used.
        node_set (str): Which set of nodes to sample from. Options:
            - 'test' (default): Test set nodes only
            - 'train': Training set nodes only
            - 'val': Validation set nodes only
            - 'all': All nodes in the dataset
            - 'train+val': Training and validation nodes
            - 'val+test': Validation and test nodes
            - 'test_has_test_nbr': Test nodes that have ≥1 1-hop neighbor also in the test set

    Returns:
        list: Indices of sampled nodes.
    """

    np.random.seed(42)
    
    # 根据 node_set 参数选择节点
    if node_set == 'test':
        available_indices = np.where(data.test_mask.numpy())[0]
    elif node_set == 'train':
        available_indices = np.where(data.train_mask.numpy())[0]
    elif node_set == 'val':
        available_indices = np.where(data.val_mask.numpy())[0]
    elif node_set == 'all':
        available_indices = np.arange(data.num_nodes)
    elif node_set == 'train+val':
        train_indices = np.where(data.train_mask.numpy())[0]
        val_indices = np.where(data.val_mask.numpy())[0]
        available_indices = np.concatenate([train_indices, val_indices])
    elif node_set == 'val+test':
        val_indices = np.where(data.val_mask.numpy())[0]
        test_indices = np.where(data.test_mask.numpy())[0]
        available_indices = np.concatenate([val_indices, test_indices])
    elif node_set == 'test_has_test_nbr':
        # 测试集中，1阶邻居中至少有一个也在测试集内的节点子集
        test_mask_np = data.test_mask.numpy()
        test_set = set(np.where(test_mask_np)[0])
        edge_np = data.edge_index.numpy()  # shape (2, E)
        # 构建无向邻接表（合并正向和反向边）
        from collections import defaultdict
        nbr = defaultdict(set)
        for s, d in zip(edge_np[0], edge_np[1]):
            nbr[s].add(d)
            nbr[d].add(s)
        # 保留在测试集中、且至少有一个测试集邻居的节点
        qualified = sorted(
            node for node in test_set
            if nbr[node] & test_set          # 与测试集的交集非空
        )
        available_indices = np.array(qualified, dtype=np.int64)
        print(f"  test_has_test_nbr: {len(available_indices)}/{len(test_set)} test nodes "
              f"have ≥1 test-set neighbor")
    elif node_set == 'test_and_1hop':
        # 测试集节点 ∪ 测试集所有1阶邻居（无向）
        test_mask_np = data.test_mask.numpy()
        test_set = set(np.where(test_mask_np)[0])
        edge_np = data.edge_index.numpy()  # shape (2, E)
        # 构建无向邻接表（合并正向和反向边）
        from collections import defaultdict
        nbr = defaultdict(set)
        for s, d in zip(edge_np[0], edge_np[1]):
            nbr[s].add(d)
            nbr[d].add(s)
        # 收集测试集所有1阶邻居
        all_neighbors = set()
        for node in test_set:
            all_neighbors.update(nbr[node])
        expanded = sorted(test_set | all_neighbors)
        available_indices = np.array(expanded, dtype=np.int64)
        non_test_nbr = len(all_neighbors - test_set)
        print(f"  test_and_1hop: {len(test_set)} test nodes + {non_test_nbr} non-test 1-hop neighbors "
              f"= {len(available_indices)} nodes total")
    else:
        raise ValueError(f"Invalid node_set '{node_set}'. Must be one of: "
                         f"'test', 'train', 'val', 'all', 'train+val', 'val+test', "
                         f"'test_has_test_nbr', 'test_and_1hop'")
    
    print(f"Available nodes in '{node_set}' set: {len(available_indices)}")
    
    # 如果 sample_size 为 None，使用所有可用节点且不打乱顺序
    if sample_size is None:
        print(f"Using all {len(available_indices)} available nodes without sampling")
        sampled_indices = available_indices.tolist()
        return sampled_indices
    
    # 如果 sample_size 大于可用节点数，使用所有可用节点
    if sample_size > len(available_indices):
        print(f"Using all {len(available_indices)} available nodes (requested {sample_size})")
        sample_size = len(available_indices)

    if dataset != "product":
        sampled_indices = np.random.choice(available_indices, size=sample_size, replace=False)
        sampled_indices = sampled_indices.tolist()

    else:
        # Sample 2 times the sample size
        # node_indices = sample_test_nodes(data, 2 * sample_size)
        sampled_indices_double = np.random.choice(available_indices, size=min(2*sample_size, len(available_indices)), replace=False)

        # Filter out the indices of nodes with title "NA\n"
        sampled_indices = [node_idx for i, node_idx in enumerate(sampled_indices_double) 
                    if text['title'][node_idx] != "NA\n"]
        sampled_indices = sampled_indices[:sample_size]

        # sanity check
        count = 0
        for node_idx in sampled_indices:
            if text['title'][node_idx] == "NA\n":
                count += 1
        assert count == 0
        assert len(sampled_indices) == sample_size

    return sampled_indices

import tiktoken

def count_tokens(messages,model="gpt-3.5-turbo-0125"):
    
    try:
        encoding = tiktoken.encoding_for_model(model)
    except:
        encoding = tiktoken.get_encoding('o200k_base')

    total_tokens = 0
    for message in messages:
        total_tokens += len(encoding.encode(message['content']))

    return total_tokens

class MessageTooLongError(Exception):
        """Exception raised when the total number of tokens in the messages exceeds the limit of 16,384 tokens."""
        pass

def get_completion_from_messages(messages, 
                                 model="gpt-3.5-turbo-0125", 
                                 temperature=0, max_tokens=500, key="", base_url=None):
    """
    Get completion from the OpenAI API based on the given messages.
    
    现已使用 Agent.py 中的 get_robust_completion 方法进行 LLM 交互。
    注意：key 和 base_url 参数已被虚置，实际配置在 Agent.py 中。

    Parameters:
        messages (list): Messages to be sent to the OpenAI API.
        model (str, optional): The name of the model to be used. Default is "gpt-3.5-turbo".
        temperature (float, optional): Sampling temperature. Default is 0.
        max_tokens (int, optional): Maximum number of tokens for the response. Default is 500.
        key (str, optional): API key (已虚置，Agent 类内部配置).
        base_url (str, optional): API base URL (已虚置，Agent 类内部配置).

    Returns:
        str: The content of the completion message.
    """
    # 使用全局 Agent 实例进行交互
    # Agent 类内部已经配置了 API key 和 base_url
    
    # 设置 Agent 的模型（如果需要）
    if model:
        global_agent.model = model
    if key and not key.startswith("sk-proj-1234567890"):
        os.environ["OPENAI_API_KEY"] = key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url

    # 使用 Agent 的 get_robust_completion 方法
    # 这个方法具有更强的错误处理能力，包括超时控制和重试机制
    try:
        response = global_agent.get_robust_completion(
            messages=messages,
            description="LLM API call",
            min_length=10,
            max_retries=10,
            temperature=temperature
        )
        return response
    except Exception as e:
        # 如果 Agent 方法失败，抛出更明确的错误
        print(f"Agent get_robust_completion failed: {e}")
        raise RuntimeError(f"Failed to get completion from Agent: {e}")


def generate_pickle_filename(dataset, hop, iteration, just_reflection, use_instructions, mode, anonymize_edges, webkb_full_abs,
                           refining=False, model=None, rewiring=False):
    """
    生成规范化的 pickle 文件名。
    
    Parameters:
        dataset (str): 数据集名称 (e.g., 'actor', 'cora', 'arxiv_2023')
        hop (int): 跳数 (1 or 2)
        iteration (int): 迭代次数 (1 or 2)
        just_reflection (bool): 是否仅使用反思
        use_instructions (bool): 是否使用指导
        mode (str): 模式 ('ego' or 'neighbors')
        refining (bool, optional): 是否使用精炼策略
        model (str, optional): 模型名称（可选）
    
    Returns:
        str: 规范化的文件名
    
    Examples:
        >>> generate_pickle_filename('actor', 1, 1, False, True, 'neighbors')
        'actor_hop1_iter1_neighbors_instr.pkl'
        
        >>> generate_pickle_filename('cora', 2, 2, True, False, 'ego', refining=True)
        'cora_hop2_iter2_ego_refl_refine.pkl'
    """
    # 基础部分：数据集_跳数_迭代
    filename_parts = [
        dataset,
        f"hop{hop}",
        f"iter{iteration}"
    ]
    
    # 模式
    filename_parts.append(mode)
    
    # 可选标志
    flags = []
    if use_instructions:
        flags.append("instr")
    if just_reflection:
        flags.append("refl")
    if refining:
        flags.append("refine")
    if anonymize_edges:
        flags.append("anon")
    if webkb_full_abs:
        flags.append("full_abs")
    if flags:
        filename_parts.extend(flags)
    
    # 如果有模型名称，添加简化的模型名
    if model:
        # 简化模型名称
        model_short = model.split('/')[-1].replace('-', '_')[:15]  # 取最后部分，最多15字符
        filename_parts.append(model_short)
    
    if rewiring:
        filename_parts.append("rewired")
    
    # 组合成文件名
    filename = "_".join(filename_parts) + ".pkl"
    
    return filename


def map_arxiv_labels(data, text, source, arxiv_style):
    """
    Map arXiv labels based on the given source and mapping style.

    Parameters:
        data: Graph data object.
        text: Textual information associated with nodes.
        source (str): Data source, e.g., "arxiv".
        arxiv_style (str): Style of arXiv label mapping, either "identifier" or "natural language".

    Returns:
        Updated text information with new labels.
    """

    if source == "arxiv":
        if arxiv_style == "identifier":
            for i in range(len(data.y)):
                text['label'][i] = "arxiv " + text['label'][i].lower()
        elif arxiv_style == "natural language":
            for i in range(len(data.y)):
                text['label'][i] = arxiv_natural_lang_mapping[text['label'][i]]
    return text



def build_multichannel_judgement(channel_pkls, data, text, dataset='actor',
                                  node_set='test_and_1hop', options=None,
                                  show_consensus=True):
    """
    从多个 iter1 pkl 文件构建多 channel 的 initial_judgement，供 iter2 使用。

    每个 channel pkl 按 node_set 重建 node_index_list，将 wrong_reason / results
    映射到全图索引数组（size = data.num_nodes），缺失节点填 None。
    最终将各 channel 的 reasoning 拼接为单一字符串，results 取多数票。

    Parameters
    ----------
    channel_pkls : list[str]
        各 channel 的 pkl 文件路径列表。
    data : PyG Data
        图数据（提供 num_nodes 等信息）。
    text : dict
        节点文本信息（包含 'label'）。
    dataset : str
        数据集名称（默认 'actor'）。
    node_set : str
        各 channel pkl 生成时使用的节点集（默认 'test_and_1hop'）。
    options : set or None
        有效类别集合，用于从 result 文本中提取类别名。
    show_consensus : bool
        是否在各 channel 推理前插入共识摘要行（默认 True）。
        设为 False 时，仅显示各 channel 的推理块，不显示 Consensus 行。

    Returns
    -------
    list : [combined_reason, combined_result]
        combined_reason[i] : 节点 i 的拼接推理字符串（None 表示无任何 channel 覆盖）
        combined_result[i] : 多数票结果字符串（None 表示无覆盖）
    """
    import pickle as _pkl

    num_nodes = data.num_nodes
    n_ch = len(channel_pkls)

    # 每个 channel 建立全图索引数组
    ch_reasons  = []   # list of list[str|None], 每个长 num_nodes
    ch_results  = []   # list of list[str|None]

    for ch_idx, pkl_path in enumerate(channel_pkls):
        with open(pkl_path, 'rb') as f:
            ch_data = _pkl.load(f)

        wr  = ch_data.get('wrong_reason', [None] * len(ch_data['results']))
        res = ch_data['results']

        # 重建 node_index_list（与生成 pkl 时一致）
        nil = sample_test_nodes(data, text, None, dataset, node_set=node_set)
        assert len(nil) == len(res), (
            f'Channel {ch_idx} ({pkl_path}): nil len {len(nil)} != results len {len(res)}'
        )

        full_reason = [None] * num_nodes
        full_result = [None] * num_nodes
        for pos, nidx in enumerate(nil):
            full_reason[nidx] = wr[pos]
            full_result[nidx] = res[pos]

        ch_reasons.append(full_reason)
        ch_results.append(full_result)
        print(f'  Channel {ch_idx+1}/{n_ch} loaded: {pkl_path}')

    # ── 合并 reasoning ──────────────────────────────────────────────
    def _extract_cat(pred, opts):
        """从预测文本中提取类别名（最早出现的有效类别）。"""
        if not pred or not opts:
            return pred
        pl = pred.lower()
        m, e = '', len(pl)
        for o in opts:
            p = pl.find(o.lower())
            if p != -1 and p < e:
                m, e = o, p
        return m if m else pred

    from collections import Counter

    combined_reason = [None] * num_nodes
    combined_result = [None] * num_nodes

    SEP = '─' * 56   # 分隔线，不含 '#' 避免被 markdown 替换逻辑影响

    for nidx in range(num_nodes):
        parts  = []   # 各 channel 格式化后的段落
        votes  = []   # 各 channel 提取出的类别（用于多数票）

        for ch_idx in range(n_ch):
            r = ch_reasons[ch_idx][nidx]
            v = ch_results[ch_idx][nidx]
            if r is None:
                continue
            cat = _extract_cat(v, options) if v is not None else '(unknown)'
            header = f'[Channel {ch_idx+1}/{n_ch}]  Predicted: {cat}'
            parts.append(f'{SEP}\n{header}\n{SEP}\n{r}')
            votes.append(cat)

        if not parts:
            continue

        # 多数票；平票时取最小 channel 编号的结果
        vote_counts   = Counter(votes)
        majority_cat  = vote_counts.most_common(1)[0][0]
        majority_cnt  = vote_counts[majority_cat]
        agreement_str = f'{majority_cnt}/{len(votes)} channels'

        # combined_reason: 带清晰层次的完整多 channel 推理块
        consensus_line = (
            f'Consensus: {majority_cat}  ({agreement_str})\n\n'
            if show_consensus else ''
        )
        combined_reason[nidx] = (
            consensus_line
            + '\n\n'.join(parts)
            + f'\n\n{SEP}'
        )
        # combined_result 置 None：让 get_node_info / handle_standard_neighbors_v2
        # 走 "## Multi-channel initial categorization and reasons ##" 分支，
        # 避免对 majority_cat 做额外的数字重编码（会破坏括号中的投票信息）。
        # Consensus 已嵌入 combined_reason 顶部，无需额外的 result 字段。
        combined_result[nidx] = None

    print(f'build_multichannel_judgement: {n_ch} channels, '
          f'{sum(1 for r in combined_reason if r is not None)} nodes covered.')
    # combined_result 全为 None，返回 None 使上层走 multi-channel 渲染分支，
    # 避免 None.replace() 错误（上层条件是 `if initial_result is not None:`）
    return [combined_reason, None]


def get_important_neighbors(node_index, neighbors, text, dataset, max_papers_1=5, k=5, key="", base_url=None, model="gpt-3.5-turbo-0125"):
    """
    Get indices of important neighboring nodes.

    Parameters:
        node_index (int): Index of the target node.
        neighbors (list): List of neighboring node indices.
        text: Textual information associated with nodes.
        dataset (str): The name of the dataset being used.
        max_papers_1 (int, optional): Maximum number of neighbor papers for the first hop. Default is 5.
        k (int, optional): Number of most important neighbors to return. Default is 5.

    Returns:
        list: Indices of important neighbors.
    """

    # Get the title of the target node
    if dataset == "actor":
        target_title = text['node_text'][node_index].split('node name:')[1].split(';')[0].strip() if 'node name:' in text['node_text'][node_index] else f"Actor {node_index}"
    else:
        target_title = text['title'][node_index]

    # Set Target_word based on dataset
    if dataset == "product":
        Target_word = "Product"
    elif dataset == "actor":
        Target_word = "Actor"
    elif dataset in ["texas", "wisconsin", "washington", "cornell"]:
        Target_word = "Webpage"
    else:
        Target_word = "Paper"

    # Create a message to ask the model for the most important papers
    message = {'role': 'system', 'content': f'The {Target_word.lower()} of interest is "{target_title}". Please return a Python list of at most {k} indices of the most related {Target_word.lower()}s among the following neighbors, ordered from most related to least related. If there are fewer than {k} neighbors, just rank the neighbors by relevance. The list should look like this: [1, 2, 3, ...]'}

    # Limit the number of neighbors based on max_papers_1
    limited_neighbors = neighbors[:max_papers_1]

    # Add the titles of each neighbor to the message
    idx_to_neighbor = {}
    for i, neighbor_idx in enumerate(limited_neighbors, start=1):
        neighbor_title = text['title'][neighbor_idx]
        message['content'] += f"\n{i}: {neighbor_title}"
        idx_to_neighbor[i] = neighbor_idx

    message['content'] += "\n\nAnswer:\n\n"

    print(f"Message: {message['content']}")

    response = get_completion_from_messages([message], key=key, base_url=base_url, model=model)

    print(f"Response: {response}")

    # Assume the model's response is a Python list of indices of the most important neighbors
    # Extract these indices from the response
    try:
        important_neighbors_indices = [idx_to_neighbor[idx] for idx in eval(response) if idx in idx_to_neighbor]
    except:
        print("Unable to parse the response as a Python list.")
        return []

    print(f"Important neighbors indices: {important_neighbors_indices}")

    return important_neighbors_indices


def handle_important_neighbors(node_index, text, dataset, all_hops, data, abstract_len, include_label, max_papers_1, key="", base_url=None):
    """
    Handle important neighbors when attention is used.

    Parameters:
        node_index (int): Index of the target node.
        text: Textual information of the node.
        dataset (str): The name of the dataset.
        all_hops (list): List of all neighbor nodes up to a certain hop.
        data: Graph data object.
        abstract_len (int): Length of the abstract to consider.
        include_label (bool): Whether to include labels.
        max_papers_1 (int): Maximum number of papers for the first hop.

    Returns:
        str: String containing information about important neighbors.
    """

    prompt_str = ""
    # Set Target_word based on dataset
    if dataset == "product":
        Target_word = "Product"
    elif dataset == "actor":
        Target_word = "Actor"
    elif dataset in ["texas", "wisconsin", "washington", "cornell"]:
        Target_word = "Webpage"
    else:
        Target_word = "Paper"
    k = 5
    attention_dir = f"attention/{dataset}/attention_{k}"
    filename = f"{attention_dir}/{node_index}.json"
    
    if os.path.exists(filename):
        with open(filename, "r") as f:
            important_neighbors = json.load(f)
    else:
        neighbors = list(set(all_hops[0]))
        important_neighbors = get_important_neighbors(node_index, neighbors, text, max_papers_1, k, key=key, base_url=base_url)
        important_neighbors = [int(x) for x in important_neighbors]
        
        if not os.path.exists(attention_dir):
            os.makedirs(attention_dir)
        with open(filename, "w") as f:
            json.dump(important_neighbors, f)

    if len(important_neighbors) > 0:
        prompt_str += f"It has following important neighbors, from most related to least related:\n"
        for i, neighbor_idx in enumerate(important_neighbors):
            neighbor_title = text['title'][neighbor_idx]
            prompt_str += f"{Target_word} {i+1} title: {neighbor_title}\n"
            if abstract_len > 0:
                neighbor_abstract = text['abs'][neighbor_idx]
                prompt_str += f"{Target_word} {i+1} abstract: {neighbor_abstract[:abstract_len]}\n"
            if include_label and (data.train_mask[neighbor_idx] or data.val_mask[neighbor_idx]):
                label = text['label'][neighbor_idx]
                prompt_str += f"Label: {label}\n"
    return prompt_str

def handle_standard_neighbors(node_index, text, all_hops, data, hop, max_papers_1,
                              max_papers_2, abstract_len, include_label, dataset, initial_judgement=None):
    """
    Handle neighbors when attention is not used.

    Parameters:
        node_index (int): Index of the target node.
        text: Textual information of the node.
        all_hops (list): List of all neighbor nodes up to a certain hop.
        data: Graph data object.
        hop (int): Number of hops to consider.
        max_papers_1 (int): Maximum number of papers for the first hop.
        max_papers_2 (int): Maximum number of papers for the second hop.
        abstract_len (int): Length of the abstract to consider.
        include_label (bool): Whether to include labels.
        dataset (str): Name of the dataset being used.

    Returns:
        str: String containing information about standard neighbors.
    """

    prompt_str = ""
    # Set Target_word based on dataset
    if dataset == "product":
        Target_word = "Product"
    elif dataset == "actor":
        Target_word = "Actor"
    elif dataset in ["texas", "wisconsin", "washington", "cornell"]:
        Target_word = "Webpage"
    else:
        Target_word = "Paper"

    for h in range(0, hop):
        neighbors_at_hop = all_hops[h]
        neighbors_at_hop = np.array(neighbors_at_hop)
        neighbors_at_hop = np.unique(neighbors_at_hop)
        if h == 0:
            neighbors_at_hop = neighbors_at_hop[:max_papers_1]
        else:
            neighbors_at_hop = neighbors_at_hop[:max_papers_2]

        if len(neighbors_at_hop) > 0:
            
            if dataset == 'product':
                prompt_str_hop = f"It has following neighbor products purchased toghther at hop {h+1}:\n"
            elif dataset == 'actor':
                prompt_str_hop = f"It has following collaborator actors at hop {h+1}:\n"
            elif dataset in ["texas", "wisconsin", "washington", "cornell"]:
                prompt_str_hop = f"It has following linked webpages at hop {h+1}:\n"
            else:
                prompt_str_hop = f"It has following neighbor papers at hop {h+1}:\n"

            #neighbors=[]
            for i, neighbor_idx in enumerate(neighbors_at_hop):

                if dataset == "actor":
                    # Actor 数据集的处理
                    neighbor_text = text['node_text'][neighbor_idx]
                    if 'node name:' in neighbor_text:
                        neighbor_name = neighbor_text.split('node name:')[1].split(';')[0].strip()
                    else:
                        neighbor_name = f"Actor {neighbor_idx}"
                    prompt_str_hop += f"{Target_word} {i+1} name: {neighbor_name}\n"
                    
                    if abstract_len > 0:
                        prompt_str_hop += f"{Target_word} {i+1} information: {neighbor_text[:abstract_len]}\n"
                else:
                    # 其他数据集的处理
                    neighbor_title = text['title'][neighbor_idx]
                    prompt_str_hop += f"{Target_word} {i+1} title: {neighbor_title}\n"
                    #neighbor={'title':neighbor_title}
                    
                    if abstract_len > 0:
                        neighbor_abstract = text['abs'][neighbor_idx]
                        prompt_str_hop += f"{Target_word} {i+1} abstract: {neighbor_abstract[:abstract_len]}\n"
                        #neighbor["abstract"]=neighbor_abstract[:abstract_len]
                
                if initial_judgement is not None: 
                    neighbor_init = initial_judgement[neighbor_idx]
                    #prompt_str_hop = f"{Target_word} {i+1} initial judgement and reason: {neighbor_init}\n" + prompt_str
                    prompt_str_hop = f"{Target_word} {i+1} initial judgement and reason: {neighbor_init}\n" + prompt_str
                    #neighbor["initial judgement and reason"]=neighbor_init

                if include_label and (data.train_mask[neighbor_idx] or data.val_mask[neighbor_idx]):
                    label = text['label'][neighbor_idx]
                    prompt_str_hop += f"Label: {label}\n"
                
                #neighbors.append(neighbor)
            prompt_str += prompt_str_hop
            
        #prompt_str += json.dumps(neighbors)+'\n'
    return prompt_str

def handle_standard_neighbors_v2(node_index, text, all_hops, data, hop, max_papers_1,
                              max_papers_2, abstract_len, include_label, dataset, abs_trunc=True, initial_judgement=None,refining=False, rewiring=False, anonymize_edges=False, webkb_full_abs=False, include_neighbors=False):
    """
    Handle standard neighbors for a specific node up to a certain hop in a directed graph.

    Parameters:
        all_hops (list): Lists of nodes for each hop distance.
        text (dict): Dictionary containing the title and abstract of each paper.
        dataset (str): Name of the dataset.
        hop (int, optional): Number of hops around the node to consider. Default is 1.
        max_papers_1 (int, optional): Maximum number of papers to consider at hop 1. Default is 20.
        max_papers_2 (int, optional): Maximum number of papers to consider at hop 2. Default is 10.
        abstract_len (int, optional): Length of the abstract to include. Default is 0.
        include_label (bool, optional): Whether to include the label of the paper. Default is False.
        data (Data, optional): Data object containing the train and validation masks.
        initial_judgement (list, optional): Initial judgement process and categorization result for each paper. Default is None.
        refining (bool, optional): Whether to use the refining style of the neighborhood description. Default is False.
        rewiring (bool, optional): Whether to rewire the neighborhood to conduct experiments on webkb datasets. Default is False.
        anonymize_edges (bool, optional): Whether to anonymize edge names (e.g., "citation" → "neighbor"). Default is False.
        webkb_full_abs (bool, optional): (WebKB only) Show full content abstract for ALL neighbor nodes,
            not just "other"-typed ones. Default is False.
        include_neighbors (bool, optional): (WebKB only) When True and hop==2, additionally append
            the second-hop link-pattern summary for each first-hop neighbor.
            Only takes effect when this flag is True; otherwise always False. Default is False.

    Returns:
        str: Prompt string.
    """
    if not (initial_judgement is None or initial_judgement[0] is None):
        initial_result=initial_judgement[1]
        initial_judgement=initial_judgement[0]
    else:
        initial_judgement=None
    #used to control whether to use the "flipping" or "extreme" style of the neighborhood description for webkb. 
    # include_neighbors 由外部参数决定；只有外部传入 True 时才可能被激活。
    prompt_str = ""
    if dataset in ["cora", "arxiv_2023"]:
        Target_word = "Paper"
    elif dataset in ["texas", "wisconsin", "washington", "cornell"]:
        Target_word = "Webpage"
        random_reconnect = False
    elif dataset == "actor":
        Target_word = "Actor"
    if hop ==2 and dataset in ["texas", "wisconsin", "washington", "cornell"]:
        hop=1
        # include_neighbors 保持外部传入的值，不再自动置 True
        if rewiring:
            random_reconnect = True
    for h in range(0, hop):
        if h == 0:
            neighbors_at_hop = all_hops[:2]
        elif h == 1:
            neighbors_at_hop = all_hops[2:6]
        else:
            neighbors_at_hop = all_hops[2*h+2:2*h+4]

        for i, direction in enumerate(neighbors_at_hop):
            if len(direction) == 0:
                continue
            
            # you can use the following code to generate the neighborhood description for direct considering high-order neighborhood in original style of (Huang et al. 2024) but with semantic of each edge in constraction to TAPTN style just for experiments. But it's abandoned for TAPTN iteration. 
            direction_str = ""
            if h == 0:
                direction_str = "directly cited by" if i % 2 == 1 else "directly cites"
            elif h == 1:
                if i % 4 == 0:
                    direction_str = "indirectly cites"
                elif i % 4 == 2:
                    direction_str = "co-citation coupled with"
                elif i % 4 == 1:
                    direction_str = "bibliographic coupled with"
                else:
                    direction_str = "indirectly cited by"
            else:
                direction_str = "indirectly cited by" if i % 2 == 1 else "indirectly cites"
            if h==1 and i % 4 == 0 or h != 1 and i % 2 == 0:
                prompt_str_head = 'It '
            else:
                prompt_str_head = 'It was '
            prompt_str_head += f"{direction_str} following papers"

            if h == 0 or h == 1 and (i % 4 == 2 or i % 4 == 1):
                prompt_str_head += ":\n"
            else:
                prompt_str_head += f" at hop {h+1}:\n"
            
            # TAPTN style of neighborhood description (for cora and arxiv-2023)
            if h == 0:
                if dataset in ["cora", "arxiv_2023"]:
                    if anonymize_edges:
                        # 匿名化边名称
                        if refining:
                            prompt_str_head = "\n#### Neighbors ####\n\nThe paper has following neighbors:\n"
                        else:
                            prompt_str_head = "\nIt has following neighbors:\n"
                    else:
                        if refining:
                            prompt_str_head = "\n#### References ####\n\nThe paper has following references:\n" if i % 2 == 0 else "\n#### Citations ####\n\nThe paper has following citations:\n"
                        else:
                            prompt_str_head = "\nIt has following references:\n" if i % 2 == 0 else "\nIt has following citations:\n"
                elif dataset in ["texas", "wisconsin", "washington", "cornell"]:
                    if anonymize_edges:
                        # 匿名化边名称
                        prompt_str_head = "\nIt has following neighbors:\n"
                    else:
                        prompt_str_head = "\nIt has outbound links to following webpages:\n" if i % 2 == 1 else "\nIt has inbound links from following webpages:\n"
                elif dataset == "actor":
                    if anonymize_edges:
                        # 匿名化边名称
                        if refining:
                            prompt_str_head = "\n#### Neighbors ####\n\nThe actor has following neighbors:\n"
                        else:
                            prompt_str_head = "\nThe actor has following neighbors:\n"
                    else:
                        if refining:
                            prompt_str_head = "\n#### Collaborators ####\n\nThe actor has collaborated with following actors:\n"
                        else:
                            prompt_str_head = "\nThe actor has collaborated with following actors:\n"
        
            #trunc reference list
            if h==0:
                if dataset in ["texas", "wisconsin", "washington", "cornell"]:
                    np.random.seed(42)
                    if len(direction)>max_papers_1:
                        direction = np.random.choice(direction, max_papers_1, replace=False)
                else:
                    direction=direction[:max_papers_1]
            else:
                direction=direction[:max_papers_2]
            if anonymize_edges:
                # 匿名化边名称
                if dataset in ["texas", "wisconsin", "washington", "cornell"]:
                    Target_word = "It has link to webpage "
                else:
                    Target_word = "neighbor"
            else:
                if dataset in ["cora","arxiv_2023"]:
                    Target_word = "reference" if i % 2 == 0 else "citation"
                elif dataset in ["texas", "wisconsin", "washington", "cornell"]:
                    Target_word = "outgoing hyperlink" if i % 2 == 0 else "incoming hyperlink"
                    #Target_word = "links to webpage" if i % 2 == 0 else "linked from webpage"
                    #Target_word = "It has outbound link to webpage OP" if i % 2 == 1 else "It has inbound link from webpage IP"
                    Target_word = "It has outbound link to webpage " if i % 2 == 1 else "It has inbound link from webpage "
                elif dataset == "actor":
                    Target_word = "collaborator"
            prompt_str_body = ""
            
            outbound_neighbors_dict={}
            inbound_neighbors_dict={}
            if dataset in ["texas", "wisconsin", "washington", "cornell"]:
                for j, neighbor_idx in enumerate(direction):
                    neighbors = get_subgraph2(neighbor_idx, data.edge_index, hop=1)
                    outbound_neighbors = neighbors[0]
                    inbound_neighbors = neighbors[1]
                    outbound_neighbors_dict[neighbor_idx]=outbound_neighbors
                    inbound_neighbors_dict[neighbor_idx]=inbound_neighbors
                # random_reconnect = False
                random.seed(42)
                if random_reconnect: 
                    # Extract and merge all outbound neighbors
                    all_outbound_neighbors = []
                    for neighbors in outbound_neighbors_dict.values():
                        all_outbound_neighbors.extend(list(set(neighbors)-set([node_index])))
                    all_inbound_neighbors = []
                    for neighbors in inbound_neighbors_dict.values():
                        all_inbound_neighbors.extend(list(set(neighbors)-set([node_index])))
                    
                    # Shuffle the merged list of outbound neighbors
                    random.shuffle(all_outbound_neighbors)
                    random.shuffle(all_inbound_neighbors)
                    
                    # Reassign the shuffled neighbors back to the keys
                    new_outbound_neighbors_dict = {}
                    num_neighbors = int(len(all_outbound_neighbors)/len(outbound_neighbors_dict.keys()))
                    for key in outbound_neighbors_dict.keys():    
                        new_outbound_neighbors_dict[key] = all_outbound_neighbors[:num_neighbors]
                        all_outbound_neighbors = all_outbound_neighbors[num_neighbors:]
                    new_inbound_neighbors_dict = {}
                    num_neighbors = int(len(all_inbound_neighbors)/len(inbound_neighbors_dict.keys()))
                    for key in inbound_neighbors_dict.keys():    
                        new_inbound_neighbors_dict[key] = all_inbound_neighbors[:num_neighbors]
                        all_inbound_neighbors = all_inbound_neighbors[num_neighbors:]
                else:
                    new_inbound_neighbors_dict = inbound_neighbors_dict
                    new_outbound_neighbors_dict = outbound_neighbors_dict
                    
            prompt_str += prompt_str_head
            prompt_str_body = ""
            for j, neighbor_idx in enumerate(direction):
                #prompt_str_body = ""
                if dataset in ["cora", "arxiv_2023"]:
                    if abs_trunc or h>=1:
                        if abstract_len > 0:
                            neighbor_abstract = text['abs'][neighbor_idx]
                            if refining:
                                prompt_str_body += f"{Target_word} {j+1} citation number:\n{data.in_degree[neighbor_idx]}\n"
                                prompt_str_body += f"{Target_word} {j+1} references number:\n{data.out_degree[neighbor_idx]}\n"
                                if not isinstance(neighbor_abstract, str):
                                    neighbor_abstract = "No abstract available."
                                prompt_str_body += f"## {Target_word} {j+1} abstract ##\n{neighbor_abstract[:abstract_len]}\n"
                            else:
                                if isinstance(neighbor_abstract, str):
                                    prompt_str_body += f"{Target_word} {j+1} abstract: {neighbor_abstract[:abstract_len]}\n"
                    else:
                        neighbor_abstract = text['abs'][neighbor_idx]
                        prompt_str_body += f"##{Target_word} {j+1} abstract##\n{neighbor_abstract}\n"

                elif dataset == "actor":
                    # Actor 数据集的处理
                    if abs_trunc or h>=1:
                        if abstract_len > 0:
                            neighbor_text = text['node_text'][neighbor_idx]
                            if refining:
                                prompt_str_body += f"{Target_word} {j+1} collaboration count:\n{data.in_degree[neighbor_idx] + data.out_degree[neighbor_idx]}\n"
                                prompt_str_body += f"## {Target_word} {j+1} information ##\n{neighbor_text[:abstract_len]}\n"
                            else:
                                prompt_str_body += f"{Target_word} {j+1} information: {neighbor_text[:abstract_len]}\n"
                    else:
                        neighbor_text = text['node_text'][neighbor_idx]
                        prompt_str_body += f"## {Target_word} {j+1} information ##\n{neighbor_text}\n"
                
                elif dataset in ["texas", "wisconsin", "washington", "cornell"]:
                    outbound_neighbors = new_outbound_neighbors_dict[neighbor_idx]
                    inbound_neighbors = new_inbound_neighbors_dict[neighbor_idx]
                    # Count the occurrences of each category in outbound and inbound neighbors
                    outbound_categories = {}
                    inbound_categories = {}
                    
                    for neighbor in outbound_neighbors:
                        category = text['label'][neighbor]
                        if category in outbound_categories:
                            outbound_categories[category] += 1
                        else:
                            outbound_categories[category] = 1
                    
                    for neighbor in inbound_neighbors:
                        category = text['label'][neighbor]
                        if category in inbound_categories:
                            inbound_categories[category] += 1
                        else:
                            inbound_categories[category] = 1
                     # Summarize the categories of inbound and outbound neighbors

                    outbound_summary = "Outbound links by category:\n" + ("\n".join([f"{cat}: {count}" for cat, count in outbound_categories.items()]) if len(outbound_categories) > 0 else "No outbound neighbors")
                    inbound_summary = "Inbound links by category:\n" + ("\n".join([f"{cat}: {count}" for cat, count in inbound_categories.items()]) if len(inbound_categories) > 0 else "No inbound neighbors")
                    if len(outbound_categories) == 0 and len(inbound_categories) == 0:
                        neighbor_summary='Private resource'
                    else:
                        neighbor_summary=outbound_summary+'\n'+inbound_summary  
                    abs_word = "words with top 100 tf-idf"
                    abs_word="content abstract"
                    #abs_word="category"
                    if abs_trunc:
                        if abstract_len > 0:
                            #neighbor_abstract = text['abs'][neighbor_idx]
                            neighbor_abstract = text['label'][neighbor_idx]
                            if neighbor_abstract == "other":
                                neighbor_abstract += f", {text['abs'][neighbor_idx]}" 
                            #prompt_str_body += f"{Target_word} {j+1} {abs_word}: {neighbor_abstract[:abstract_len]}\n"
                            neighbor_label = text['label'][neighbor_idx]
                            show_abs = webkb_full_abs or neighbor_label == 'other'
                            if webkb_full_abs:
                                # 全摘要模式：不显示标签，直接附加内容摘要
                                prompt_str_body += (
                                    f"{Target_word}{text['title'][neighbor_idx]}"
                                    + (f" with content abstract as below: {text['abs'][neighbor_idx][:abstract_len]}\n"
                                       if show_abs else '\n')
                                )
                            else:
                                prompt_str_body += (
                                    f"{Target_word}{text['title'][neighbor_idx]} which is a {neighbor_label} page"
                                    + (f" with content abstract as below: {text['abs'][neighbor_idx][:abstract_len]}\n"
                                       if show_abs else '\n')
                                )
                            if include_neighbors:
                                prompt_str_body += ", with link patterns as below:\n"
                                prompt_str_body += f"{neighbor_summary}\n"
                    else:
                        neighbor_abstract = text['abs'][neighbor_idx]
                        prompt_str_body += f"{Target_word} {j+1} {abs_word}: {neighbor_abstract}\n"

                if dataset in ["cora","arxiv_2023"]:
                    neighbor_title = text['title'][neighbor_idx]
                    if refining:
                        prompt_str_body += f"## {Target_word} {j+1} title ##\n{neighbor_title}\n"
                    else:
                        prompt_str_body += f"{Target_word} {j+1} title: {neighbor_title}\n"
                elif dataset == "actor":
                    # Actor 数据集从 node_text 中提取名字
                    node_text = text['node_text'][neighbor_idx]
                    # 提取演员名字 (格式: "node name: XXX; ...")
                    if 'node name:' in node_text:
                        neighbor_name = node_text.split('node name:')[1].split(';')[0].strip()
                    else:
                        neighbor_name = f"Actor {neighbor_idx}"
                    
                    if refining:
                        prompt_str_body += f"## {Target_word} {j+1} name ##\n{neighbor_name}\n"
                    else:
                        prompt_str_body += f"{Target_word} {j+1} name: {neighbor_name}\n"
                
                # elif dataset == "wisconsin":    
                #     prompt_str_body += f"{Target_word} {j+1} URL: {neighbor_title}\n"
                
                if initial_judgement is not None: 
                    neighbor_init = initial_judgement[neighbor_idx]
                    # 若邻居节点无初始推理（如训练/验证节点），跳过该部分
                    if neighbor_init:
                        if initial_result is not None:
                            neighbor_result = initial_result[neighbor_idx].replace('\n',',')
                            # Remove 'number.' pattern
                            neighbor_result = re.sub(r'\d+\.\s*', '', neighbor_result)
                            neighbor_result = '; '.join([f"{cn+1}. {cl}" for cn,cl in enumerate(neighbor_result.split(','))])
                            # Assuming target_init is a multiline string
                        lines = neighbor_init.split('\n')
                        for il, line in enumerate(lines):
                            match = re.match(r'^(#+)(.*)$', line)
                            if match:
                                # Replace '#' with '**' and add '**' at the end
                                lines[il] = '**' + match.group(2).replace('#','').strip() + '**'

                        # Join the lines back into a single string
                        neighbor_init = '\n'.join(lines)
                        # WebKB + webkb_full_abs 模式：使用 URL 标题作为章节标识（旧版 wisconsin 格式）
                        _webkb_title_fmt = (
                            webkb_full_abs
                            and dataset in ["texas", "wisconsin", "washington", "cornell"]
                        )
                        if initial_result is not None:
                            if _webkb_title_fmt:
                                _nb_title = text['title'][neighbor_idx]
                                prompt_str_body = (
                                    f"## Initial categorization for {_nb_title}: ##\n{neighbor_result}\n"
                                    f"## Reasons of initial categorization for {_nb_title}: ##\n{neighbor_init}\n\n"
                                ) + prompt_str_body
                            else:
                                prompt_str_body = f"## {Target_word} {j+1} initial categorization ##\n{neighbor_result}\n## {Target_word} {j+1} reasons for initial categorization ##\n{neighbor_init}\n\n" + prompt_str_body
                        else:
                            if _webkb_title_fmt:
                                _nb_title = text['title'][neighbor_idx]
                                prompt_str_body = (
                                    f"## Initial categorization and reasons for {_nb_title}: ##\n{neighbor_init}\n\n"
                                ) + prompt_str_body
                            else:
                                # multi-channel 模式：combined_result 为 None，
                                # combined_reason 已含 Consensus 行及各 channel 块
                                prompt_str_body = (
                                    f"## {Target_word} {j+1} multi-channel initial categorization and reasons ##\n"
                                    f"{neighbor_init}\n\n"
                                ) + prompt_str_body

                if include_label and (data.train_mask[neighbor_idx] or data.val_mask[neighbor_idx]):
                    label = text['label'][neighbor_idx]
                    prompt_str_body += f"Label: {label}\n"

                #prompt_str += f"\n### {Target_word} {j+1} ###\n\n"+prompt_str_body
            prompt_str += prompt_str_body

    return prompt_str

def get_node_info(node_indices, data, text, mode, dataset, source, hop=1, max_papers_1=20, max_papers_2=10, 
                  abstract_len=0, print_prompt=True, include_label=False, return_message=False, 
                  arxiv_style=False, include_options=False, include_abs=False, zero_shot_CoT=False, 
                  few_shot=False, use_attention=False, explain=False, initial_judgement=None, comfirm=False, options=None, revised_judgement=None,refining=False, key="", base_url=None, rewiring=False, just_reflection=False, use_instructions=False, first_iter=False, anonymize_edges=False, webkb_full_abs=False, include_neighbors=False):
    """
    Main function to get node information based on various modes and options.

    Parameters:
        node_indices (list): List of node indices to consider.
        data: Graph data object.
        text: Textual information associated with nodes.
        mode (str): Mode of operation, either 'neighbors' or 'ego'.
        dataset (str): Name of the dataset being used.
        source (str): Source of the data.
        hop (int, optional): Number of hops to consider. Default is 1.
        max_papers_1 (int, optional): Maximum number of papers for the first hop. Default is 20.
        max_papers_2 (int, optional): Maximum number of papers for the second hop. Default is 10.
        abstract_len (int, optional): Length of the abstract to consider. Default is 0.
        print_prompt (bool, optional): Whether to print the prompt. Default is True.
        include_label (bool, optional): Whether to include labels. Default is False.
        return_message (bool, optional): Whether to return the message. Default is False.
        arxiv_style (bool, optional): Whether to use arXiv style for labels. Default is False.
        include_options (bool, optional): Whether to include options in the system prompt. Default is False.
        include_abs (bool, optional): Whether to include abstracts. Default is False.
        zero_shot_CoT (bool, optional): Whether to use zero-shot CoT. Default is False.
        few_shot (bool, optional): Whether to use few-shot learning. Default is False.
        use_attention (bool, optional): Whether to use attention. Default is False.
        just_reflection (bool, optional): Whether to just self-reflection or use TAPTN iteration. Default is False, meaning using TAPTN style.
        anonymize_edges (bool, optional): Whether to anonymize edge names (e.g., "citation" → "neighbor"). Default is False.
        webkb_full_abs (bool, optional): (WebKB only) Show full content abstract for the target node and
            ALL neighbor nodes, not just "other"-typed ones. Default is False.
        include_neighbors (bool, optional): (WebKB only) Append second-hop link-pattern summary for each
            first-hop neighbor when hop==2. Controlled via command line; default False.

    Returns:
        Depending on the 'return_message' flag, either prints the prompt and ideal answer or returns a list of messages.
    """
    if initial_judgement is not None:
        initial_result=initial_judgement[1]
        initial_judgement=initial_judgement[0]
    else:
        initial_result=None

    for node_index in node_indices:
        
        if mode == 'neighbors':
            # Initial setup for neighbors mode
            if dataset in ["cora","arxiv_2023"]:
                title = text['title'][node_index]
                if refining:
                    prompt_str = f"## Title ##\n{title}\n"
                else:
                    prompt_str = f"Title: {title}\n"
                
            elif dataset in ["texas", "wisconsin", "washington", "cornell"]:
                title = text['title'][node_index]
                prompt_str = f"URL: {title}\n"
            
            elif dataset == "actor":
                # Actor 数据集从 node_text 中提取名字
                node_text = text['node_text'][node_index]
                if 'node name:' in node_text:
                    actor_name = node_text.split('node name:')[1].split(';')[0].strip()
                else:
                    actor_name = f"Actor {node_index}"
                
                if refining:
                    prompt_str = f"## Actor Name ##\n{actor_name}\n"
                else:
                    prompt_str = f"Actor Name: {actor_name}\n"
            
            # Include abstract if required
            if include_abs:
                if source in ['cora' ,'arxiv']:
                    abstract = text['abs'][node_index]
                    if refining:
                        prompt_str = f"## Abstract ##\n{abstract}\n" + prompt_str
                    else:
                        prompt_str = f"Abstract: {abstract}\n" + prompt_str
                    prompt_str += f"References Number:\n{data.out_degree[node_index]}\n"
                    prompt_str += f"Citations number:\n{data.in_degree[node_index]}\n"   
                
                elif source == 'actor':
                    # Actor 数据集包含 Wikipedia 信息
                    actor_info = text['node_text'][node_index]
                    if refining:
                        prompt_str = f"## Actor Information ##\n{actor_info}\n" + prompt_str
                    else:
                        prompt_str = f"Actor Information: {actor_info}\n" + prompt_str
                    prompt_str += f"Collaboration count:\n{data.out_degree[node_index] + data.in_degree[node_index]}\n"

                elif source == 'wisconsin':
                    abs_word = "Words with top 100 tf-idf"
                    abs_word = "Content abstract"
                    abstract = text['abs'][node_index]
                    # Get outbound and inbound neighbors at hop one
                    neighbors = get_subgraph2(node_index, data.edge_index, hop=1)
                    outbound_neighbors = neighbors[0]
                    inbound_neighbors = neighbors[1]
                    # Count the occurrences of each category in outbound and inbound neighbors
                    outbound_categories = {}
                    inbound_categories = {}
                    
                    for neighbor in outbound_neighbors:
                        category = text['label'][neighbor]
                        if category in outbound_categories:
                            outbound_categories[category] += 1
                        else:
                            outbound_categories[category] = 1
                    
                    for neighbor in inbound_neighbors:
                        category = text['label'][neighbor]
                        if category in inbound_categories:
                            inbound_categories[category] += 1
                        else:
                            inbound_categories[category] = 1

                     # Summarize the categories of inbound and outbound neighbors
                    outbound_summary = "Outbound neighbors by category:\n" + ("\n".join([f"{cat}: {count}" for cat, count in outbound_categories.items()]) if len(outbound_categories) > 0 else "No outbound neighbors")
                    inbound_summary = "Inbound neighbors by category:\n" + ("\n".join([f"{cat}: {count}" for cat, count in inbound_categories.items()]) if len(inbound_categories) > 0 else "No inbound neighbors")
                    
                    prompt_str = f"##Outgoing links number ##\n{data.out_degree[node_index]}\n"+ prompt_str
                    prompt_str = f"##Incoming links number ##\n{data.in_degree[node_index]}\n"+ prompt_str
                    # 当 webkb_full_abs=True 时，为目标节点附加完整内容摘要
                    if webkb_full_abs:
                        prompt_str = f"## Content Abstract ##\n{abstract}\n" + prompt_str

            #prompt_str += f"\t<Title>{title}</Title>\n"
            if initial_judgement is not None:   
                target_init = initial_judgement[node_index]
                if initial_result is not None:
                    target_result = initial_result[node_index].replace('\n',',') 
                    target_result = re.sub(r'\d+\.\s*', '', target_result)
                    target_result = '; '.join([f"{cn+1}. {cl}" for cn,cl in enumerate(target_result.split(','))])
                    # Assuming target_init is a multiline string
                lines = target_init.split('\n')
                for il, line in enumerate(lines):
                    match = re.match(r'^(#+)(.*)$', line)
                    if match:
                        # Replace '#' with '**' and add '**' at the end
                        lines[il] = '**' + match.group(2).replace('#','').strip() + '**'

                # Join the lines back into a single string
                target_init = '\n'.join(lines)
                
                if initial_result is not None:
                    prompt_str += f"## Initial categorization ##\n{target_result}\n## Reasons for initial categorization ##\n{target_init}\n\n"
                else:
                    # multi-channel 模式：combined_result 为 None，
                    # combined_reason 已含 Consensus 行及各 channel 块
                    prompt_str += f"## Multi-channel initial categorization and reasons ##\n{target_init}\n\n"
            
            sys_prompt_str = generate_system_prompt(source, arxiv_style=arxiv_style, include_options=include_options, exlain=explain, comfirm=comfirm, options=options, use_instructions=use_instructions, first_iter=first_iter)
            all_hops = get_subgraph2(node_index, data.edge_index, 2 if hop>=2 else 1)
            
            # Check for test nodes
            if data.train_mask[node_index] or data.val_mask[node_index]:
                print('node indices should only contain test nodes!!')

            # Handle neighbors based on attention
            if use_attention:
                prompt_str += handle_important_neighbors(node_index, text, dataset, all_hops, data, abstract_len, include_label, max_papers_1, key=key, base_url=base_url)
            else:
                if just_reflection:
                    # No Message Passing
                    initial_judgement_neighbor = None
                else:
                    initial_judgement_neighbor = [initial_judgement,initial_result]
                prompt_str += handle_standard_neighbors_v2(node_index, text, all_hops, data, hop, max_papers_1, max_papers_2, 
                                                         1000, include_label, dataset, initial_judgement=initial_judgement_neighbor,refining=refining, rewiring=rewiring, anonymize_edges=anonymize_edges, webkb_full_abs=webkb_full_abs, include_neighbors=include_neighbors)
            if refining:
                if dataset in ["cora", "arxiv_2023"]:
                    prompt_str = '#### Paper ####\n'+prompt_str
                elif dataset in ["texas", "wisconsin", "washington", "cornell"]:
                    prompt_str = '#### Webpage ####\n'+prompt_str
                elif dataset == "actor":
                    prompt_str = '#### Actor ####\n'+prompt_str
            
            # Finalize prompt for neighbors mode
            if dataset in ["cora","arxiv_2023"]:
                if dataset == 'cora':
                    options=["Rule Learning", "Neural Networks", "Case Based", "Genetic Algorithms", "Computational Learning Theory", "Reinforcement Learning", "Probabilistic Methods"]
                elif dataset == 'arxiv_2023':
                    options=set([f'{key} ({arxiv_natural_lang_mapping[key]})' for key in ['cs.GT','cs.MA','cs.RO','cs.NE','cs.IR','cs.SI','cs.CY']])
                prompt_str += refining5_3+"\n\n####Question####:\nPredict the 2 most appropriate category for the paper. For each category you predict, give a relevance score from 0 and 1. Make double choices from the given list of categories:{}\n\nAnswer:\n\n".format('\n'.join(options)) if use_instructions else "####Question####:\nPredict the 2 most appropriate category for the paper. For each category you predict, give a relevance score from 0 and 1. Make double choices from the given list of categories:{}\n\nAnswer:\n\n".format('\n'.join(options))
            elif dataset in ["wisconsin", "cornell", "texas", "washington"]:
                prompt_str += "####Question####:\nPredict the 2 most appropriate categories for the webpage with URL: {}. For each category you predict, give a relevance score from 0 and 1. Make double choices from the given list of categories:{}\n\nAnswer:\n\n".format(text['title'][node_index],'\n'.join(['faculty', 'staff', 'department', 'course', 'project', 'student','other']))
            elif dataset == "actor":
                # Actor 数据集的问题
                actor_name = text['node_text'][node_index].split('node name:')[1].split(';')[0].strip() if 'node name:' in text['node_text'][node_index] else f"Actor {node_index}"
                if options is not None:
                    options_list = '\n'.join(sorted(list(options)))
                else:
                    options_list = '\n'.join(["American film actors (only)", "American film actors and American television actors", "American television actors and American stage actors", "English actors", "Canadian actors"])
                prompt_str += f"####Question####:\nPredict the most appropriate category for the actor {actor_name}. Choose from the given list of categories:\n{options_list}\n\nAnswer:\n\n" if use_instructions else f"####Question####:\nPredict the most appropriate category for the actor {actor_name}. Choose from the given list of categories:\n{options_list}\n\nAnswer:\n\n"
            if zero_shot_CoT:
                prompt_str += "Let's think step by step.\n\n"                   
                
            # Return the message
            if return_message:
                return [{'role':'system', 'content': sys_prompt_str}, {'role':'user', 'content': f"{prompt_str}"}]
        
        elif mode == 'ego':
            # Formulate the prompt
            sys_prompt_str_abs  = generate_system_prompt(source, arxiv_style, include_options=include_options, first_iter=first_iter, use_instructions=use_instructions)
            
            few_shot_examples = ""
            if few_shot:
                with open(f"few_shot_examples/{dataset}.txt", 'r') as f:
                    few_shot_examples = f.read()

            if dataset == "actor":
                # Actor 数据集处理
                actor_info = text['node_text'][node_index]
                actor_name = actor_info.split('node name:')[1].split(';')[0].strip() if 'node name:' in actor_info else f"Actor {node_index}"
                if include_abs:
                    prompt_str = f"{few_shot_examples}\nActor Information: {actor_info}\nActor Name: {actor_name}\n"
                else:
                    prompt_str = f"{few_shot_examples}\nActor Name: {actor_name}\n"
            else:
                # 原有的 paper/webpage 处理
                title = text['title'][node_index]
                abstract = text['abs'][node_index]
                if include_abs:
                    prompt_str = f"{few_shot_examples}\nAbstract: {abstract}\nTitle: {title}\n"
                else:
                    prompt_str = f"{few_shot_examples}\nTitle: {title}\n"
            if initial_judgement is not None:   
                target_init = initial_judgement[node_index]
                prompt_str += "Initial judgement and reason: "+f"{target_init}"+"\n"
            if zero_shot_CoT:
                prompt_str += "Answer: \n\n Let's think step by step.\n"
            else:
                if explain:
                    entity_type = "actor" if dataset == "actor" else "paper"
                    prompt_str += f"Please further revise your initial judgement of the most appropriate category for the {entity_type}. If multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, explain how it is present in the text.\n\nAnswer: \n\n"
                else:
                    entity_type = "actor" if dataset == "actor" else "paper"
                    prompt_str += f"Please further revise your initial judgement of the most appropriate category for the {entity_type}. Do not provide your reasoning.\nAnswer: \n\n"

            if return_message:
                return [{'role':'system', 
                        'content': sys_prompt_str_abs},    
                        {'role':'user', 
                        'content': f"{prompt_str}"}] 
            
        else:
            print('Invalid mode! Please use either "neighbors" or "abstract"')


def normalize_label(label):
    """
    规范化标签以进行容错比较。
    
    Parameters:
    - label (str): 要规范化的标签
    
    Returns:
    - str: 规范化后的标签
    """
    if label is None:
        return ""
    
    # 转换为字符串（以防万一）
    label = str(label)
    
    # 去除首尾空白
    label = label.strip()
    
    # 去除末尾的标点符号（句号、逗号、感叹号、问号等）
    while label and label[-1] in '.,!?;:':
        label = label[:-1].strip()
    
    # 转换为小写以进行不区分大小写的比较
    label = label.lower()
    
    return label


def get_matched_option(prediction, valid_options):
    """
    Extracts options from the prediction string and returns the first matched option.

    Parameters:
    - prediction (str): The prediction string containing potential options.
    - valid_options (set): The set of valid options to match against.

    Returns:
    - str: The first matched option or an empty string if no matches are found.
    """
    #prediction = prediction.replace('-', ' ')
    #prediction=prediction.title()
    prediction = prediction.lower()
    matched_option = ""
    earliest_position = len(prediction)

    # Iteratively check each substring of the prediction
    for option in valid_options:
        position = prediction.find(option.lower())
        if position != -1 and position < earliest_position:
            matched_option = option
            earliest_position = position
    if matched_option == "Computational Learning Theory":
        matched_option = "Theory"
    # Return the first matched option if available, else return an empty string
    return matched_option

def get_matched_option2(prediction, valid_options):
    """
    Extracts options from the prediction string and returns the last matched option.

    Parameters:
    - prediction (str): The prediction string containing potential options.
    - valid_options (set): The set of valid options to match against.

    Returns:
    - str: The last matched option or an empty string if no matches are found.
    """
    matched_options = []

    # Iteratively check each substring of the prediction
    for option in valid_options:
        if option in prediction:
            matched_options.append(option)

    # Return the last matched option if available, else return an empty string
    return matched_options


def print_node_info_and_compare_prediction(node_index, data, text, include_label, dataset, source, 
                                           abstract_len=0, hop=1, mode="neighbors", max_papers_1=15, 
                                           max_papers_2=5, print_out=False, print_prompt=False, arxiv_style=False, 
                                           include_options=False, include_abs=False, zero_shot_CoT=False, 
                                           few_shot=False, use_attention=False, options=None, explain=False, initial_judgement=None,api_key="",base_url = None, rewiring=False, just_reflection=False,refining=False, use_instructions=False, first_iter=False, model='gpt-3.5-turbo-0125', anonymize_edges=False, webkb_full_abs=False, include_neighbors=False):
    """
    Print node information, generate a message, and compare the generated message with the ideal answer.

    Parameters:
        node_index (int): Index of the node.
        data: Graph data object.
        text: Textual information associated with nodes.
        include_label (bool): Whether to include labels.
        dataset (str): Name of the dataset being used.
        source (str): Source of the data.
        abstract_len (int, optional): Length of the abstract. Default is 0.
        hop (int, optional): Number of hops to consider. Default is 1.
        mode (str, optional): Mode of operation, either 'neighbors' or 'ego'. Default is 'neighbors'.
        max_papers_1 (int, optional): Maximum number of papers for the first hop. Default is 15.
        max_papers_2 (int, optional): Maximum number of papers for the second hop. Default is 5.
        print_out (bool, optional): Whether to print the output. Default is False.
        print_prompt (bool, optional): Whether to print the prompt. Default is False.
        arxiv_style (bool, optional): Whether to use arXiv style for labels. Default is False.
        include_options (bool, optional): Whether to include options in the system prompt. Default is False.
        include_abs (bool, optional): Whether to include abstracts. Default is False.
        zero_shot_CoT (bool, optional): Whether to use zero-shot CoT. Default is False.
        few_shot (bool, optional): Whether to use few-shot learning. Default is False.
        use_attention (bool, optional): Whether to use attention. Default is False.
        options (set, optional): Set of valid options. Required if zero_shot_CoT is True.

    Returns:
        int: Returns 1 if the prediction is correct, otherwise 0.
    """

    message = get_node_info([node_index], data, text, hop=hop, dataset=dataset, source=source,
                            mode=mode, max_papers_1=max_papers_1, max_papers_2=max_papers_2, return_message=True, 
                            include_label=include_label, abstract_len=abstract_len, print_prompt=print_prompt,
                            arxiv_style=arxiv_style, include_options=include_options, 
                            zero_shot_CoT=zero_shot_CoT, few_shot=few_shot, include_abs=include_abs,
                            use_attention=use_attention, explain=explain, initial_judgement=initial_judgement,refining=refining, key=api_key, base_url=base_url,rewiring=rewiring, just_reflection=just_reflection,use_instructions=use_instructions, first_iter=first_iter, anonymize_edges=anonymize_edges, webkb_full_abs=webkb_full_abs, include_neighbors=include_neighbors)

    if print_out:
        print(message[0]['content'], end="\n\n")
        print(message[1]['content'], end="\n\n")

    ideal_answer = text['label'][node_index]
    
    print("Ideal_answer:", ideal_answer, end="\n\n")
    
    # Get completion message and print
    response = get_completion_from_messages(message, key=api_key,base_url=base_url, model=model)
    if print_out:
        print(response)
    
    if source == "arxiv" and arxiv_style == "identifier": 
        response = response.lower()

    prediction = response if response is not None else ""
    print("Prediction: ", prediction, end="\n\n")
    if explain:
        return prediction
    
    # Compare the prediction with ideal_answer
    options_list='\n'.join(options)
    if dataset == 'arxiv_2023':
        options_list='\n'.join([f'{key} ({arxiv_natural_lang_mapping[key]})' for key in options])
        if initial_judgement is None:
            message2=[{'role':'system', 'content': f'Please extract the refined most appropriate category for the paper. Choose from the following categories:\n\n{options_list}'}, {'role':'user', 'content': f"{prediction}\n\nAnswer: \n\n"}] 
        else:    
            message2=[{'role':'system', 'content': f'Please extract the most appropriate category for the paper. Choose from the following categories:\n\n{options_list}\n\n Format your answer as "Category: [category] (Reason: [reason])". Your predicted category must strcitly the same as one of the categories in the list.'}, {'role':'user', 'content': f"#### Initial Category and Reasons ####\n\n{initial_judgement[0][node_index]}\n\n#### Refined Category and Reasons #####\n\n{prediction}\n\nAnswer: \n\n"}] 
    elif dataset == 'cora':
        message2=[{'role':'system', 'content': f'Please extract the most appropriate category for the paper. Make single choise from the following categories:\n\n{options_list}'}, {'role':'user', 'content': f"{prediction}\n\n Format Requirements: Your answer must be formatted as \"Category: [category] (Reason: [reason])\", where the predicted category must strcitly be one of the categories in the list.\n\nAnswer: \n\n"}] 
    elif dataset in ["wisconsin", "cornell", "texas", "washington"]:
        options_list='\n'.join(['faculty', 'staff', 'department', 'course', 'project', 'student'])
        message2=[{'role':'system', 'content': f'Please extract the category with highest relevance score for the webpage. Make single choise from the following categories:\n\n{options_list}'}, {'role':'user', 'content': f"{prediction}\n\nAnswer: \n\n"}]
    elif dataset == 'actor':
        message2=[{'role':'system', 'content': f'Please extract the most appropriate category for the actor. Make single choise from the following categories:\n\n{options_list}'}, {'role':'user', 'content': f"{prediction}\n\nAnswer: \n\n"}]
    print(message2[0]['content'], end="\n\n")
    print(message2[1]['content'], end="\n\n")
    while True:
        try:
            prediction2 = get_completion_from_messages(message2,model=model,key=api_key,base_url=base_url)
            prediction3 = get_matched_option(prediction2, options)   
            if prediction3 != "":
                break
        except Exception as e:
            print(f"Error in get_completion_from_messages: {e}")
            sleep(1)
            
    if prediction2 is not None:
        print("Prediction: ", prediction2)
        prediction3 = get_matched_option(prediction2, options)   
        if prediction3 == "":
            message3=[{'role':'system', 'content': f'Please find the category that is most similar to the given category in the following list of categories:\n\n{options_list}'}, {'role':'user', 'content': f"{prediction2}\n\nAnswer: \n\n"}] 
            prediction3 = get_completion_from_messages(message3,model=model,key=api_key,base_url = base_url)
            print("Prediction: ", prediction3)
            prediction3 = get_matched_option(prediction3, options)
        # Compare the prediction with ideal_answer
        if dataset=='cora_year':
            print("Is prediction correct? ", ideal_answer in prediction2.lower(), end="\n\n")
            return int(ideal_answer in prediction2.lower()), prediction, prediction2
        print("Is prediction correct? ", prediction3 == ideal_answer, end="\n\n")
        
        return int(prediction3 == ideal_answer), prediction, prediction2
    else:
        print("No valid prediction could be made.")


def process_and_compare_predictions(node_index_list, data, text, dataset_name, source, hop=2, 
                                    max_papers_1=20, max_papers_2=10, mode="title", 
                                    include_label=True, abstract_len=0, arxiv_style=False, 
                                    include_options=False, include_abs=False, zero_shot_CoT=False, 
                                    few_shot=False, use_attention=False, options=None, timeout=6000, explain=False, initial_judgement=None, api_key="",base_url=None, rewiring=False, refining=False, just_reflection=False, use_instructions=False, first_iter=False,model='gpt-3.5-turbo-0125', max_workers=1, anonymize_edges=False, webkb_full_abs=False, include_neighbors=False):
    """
    Process and compare predictions for a list of node indices.

    Parameters:
        node_index_list (list): List of node indices to process.
        data: Graph data object.
        text: Textual information associated with nodes.
        dataset (str): Name of the dataset being used.
        source (str): Source of the data.
        hop (int, optional): Number of hops to consider. Default is 2.
        max_papers_1 (int, optional): Maximum number of papers for the first hop. Default is 20.
        max_papers_2 (int, optional): Maximum number of papers for the second hop. Default is 10.
        mode (str, optional): Mode of operation, either 'title' or other modes. Default is 'title'.
        include_label (bool, optional): Whether to include labels. Default is True.
        abstract_len (int, optional): Length of the abstract to consider. Default is 0.
        arxiv_style (bool, optional): Whether to use arXiv style for labels. Default is False.
        include_options (bool, optional): Whether to include options in the system prompt. Default is False.
        include_abs (bool, optional): Whether to include abstracts. Default is False.
        zero_shot_CoT (bool, optional): Whether to use zero-shot CoT. Default is False.
        few_shot (bool, optional): Whether to use few-shot learning. Default is False.
        use_attention (bool, optional): Whether to use attention. Default is False.
        options (set, optional): Set of valid options. Required if zero_shot_CoT is True.
        timeout (int, optional): Maximum time to wait for a function to complete. Default is 60.
        rewiring (bool, optional): Whether to rewire the neighborhood for webkb datasets. Default is False.
        max_workers (int, optional): Maximum number of concurrent workers for parallel processing. 
                                     Default is 1 (serial processing for backward compatibility).
                                     Recommended: 5-10 for concurrent processing.
        anonymize_edges (bool, optional): Whether to anonymize edge names (e.g., "citation" → "neighbor"). 
                                         Default is False. Useful for ablation studies.
        webkb_full_abs (bool, optional): (WebKB only) Show full content abstract for target node and
                                         ALL neighbor nodes. Default is False.
        include_neighbors (bool, optional): (WebKB only) Append second-hop link-pattern summary for each
                                            first-hop neighbor when hop==2. Default is False.

    Returns:
        tuple: The first element is the accuracy of the predictions, and the second is a list of wrong indexes.
    """
 
    # Initialize variables
    count = 0
    wrong_indexes = []
    # 🔧 FIX: 使用固定索引数组而不是 append
    wrong_reason = [None] * len(node_index_list)
    results = [None] * len(node_index_list)
    too_long_indexes = []
    base_sleep_time = 0.5  # Starting sleep time
    max_sleep_time = 60  # Maximum sleep time
    explainations=[]
    
    # 生成规范化的 pickle 文件名
    iteration = 1 if first_iter else 2
    pickle_filename = generate_pickle_filename(
        dataset=dataset_name,
        hop=hop,
        iteration=iteration,
        just_reflection=just_reflection,
        use_instructions=use_instructions,
        mode=mode,
        refining=refining,
        model=model,
        anonymize_edges=anonymize_edges,
        webkb_full_abs=webkb_full_abs,
        rewiring=rewiring
    )
    print(f"Progress will be saved to: {pickle_filename}")
    
    if dataset_name in ["texas", "wisconsin", "washington", "cornell"]:
        if rewiring and hop == 1:
            data.edge_index = data.edge_index[[1,0],:]
    
    # Helper function to process a single node
    def process_single_node(idx, node_index):
        """Process a single node and return results"""
        # 🔧 FIX: 不再跳过无边节点，以支持 ego 模式
        # 注释掉原来的跳过逻辑
        if data.in_degree[node_index]==0 and data.out_degree[node_index]==0:
            return None
        
        retries = 0
        while True:  # Infinite loop for retries
            result_container = [None]  # List to store the result of the threaded function
            exception_container = [None]  # List to store exceptions if any
            
            # Function to run in the thread
            def thread_target():
                try:
                    print(f"Processing index {idx} (node {node_index})...")
                    result,reason,result2 = print_node_info_and_compare_prediction(
                        node_index, data, text, dataset=dataset_name, source=source, 
                        hop=hop, max_papers_1=max_papers_1, 
                        max_papers_2=max_papers_2, mode=mode, 
                        include_label=include_label, print_out=True, 
                        arxiv_style=arxiv_style, include_options=include_options, 
                        zero_shot_CoT=zero_shot_CoT, few_shot=few_shot, include_abs=include_abs, 
                        use_attention=use_attention, options=options, explain=explain, 
                        initial_judgement=initial_judgement,api_key=api_key,base_url=base_url, 
                        rewiring=rewiring, refining=refining, just_reflection=just_reflection, 
                        use_instructions=use_instructions, first_iter=first_iter, model=model, anonymize_edges=anonymize_edges,
                        webkb_full_abs=webkb_full_abs, include_neighbors=include_neighbors)
                    result_container[0] = result
                    result_container.append(reason)
                    result_container.append(result2)
                except Exception as e:
                    exception_container[0] = e
            
            # Start the function in a separate thread
            thread = threading.Thread(target=thread_target)
            thread.start()
            thread.join(timeout=timeout)

            if exception_container[0] is MessageTooLongError:
                print(f"Message too long at index {idx}")
                result_container[0]='Genetic Algorithms'
            
            if result_container[0] is not None:
                return {
                    'idx': idx,
                    'node_index': node_index,
                    'success': True,
                    'result': result_container[0],
                    'reason': result_container[1],
                    'result2': result_container[2],
                    'explain': explain
                }
            
            # If there was an exception or timeout
            else:
                if exception_container[0]:  # If there was an exception
                    print(f"An error occurred at index {idx}: {exception_container[0]}")
                else:  # If there was a timeout
                    print(f"Function timed out at index {idx}")
                
                retries += 1
                if retries >= 5:
                    # Use initial judgment as fallback
                    fallback_result = get_matched_option(initial_judgement[1][node_index],options) if initial_judgement else None
                    return {
                        'idx': idx,
                        'node_index': node_index,
                        'success': False,
                        'result': fallback_result,
                        'reason': initial_judgement[0][node_index] if initial_judgement else 'Too long to process',
                        'result2': fallback_result,
                        'explain': explain,
                        'failed': True
                    }
                
                sleep_time = min(base_sleep_time * (2 ** retries) + randint(0, 1000) / 1000, max_sleep_time)
                print(f"Retrying in {sleep_time} seconds...")
                sleep(sleep_time)
    
    # Choose between serial and concurrent processing based on max_workers
    if max_workers == 1:
        # Serial processing (original behavior for backward compatibility)
        i = 0
        while i < len(node_index_list):
            node_result = process_single_node(i, node_index_list[i])
            if node_result is None:
                i += 1
                continue
            
            # Process the result
            if node_result.get('explain', False):
                explainations.append(node_result['result'])
            else:
                count += node_result['result']
                # 🔧 FIX: 使用索引赋值
                wrong_reason[i] = node_result['reason']
                results[i] = node_result['result2']
                print(f"Prediction: {node_result['result']}")
                if node_result['result'] == 0:  # If the prediction is wrong
                    # 🔧 FIX: 存储全局节点索引而不是测试列表位置
                    wrong_indexes.append(node_index_list[i])
            
            if node_result.get('failed', False):
                count += (node_result['result2'] == text['label'][node_index_list[i]])
            
            i += 1
            
            if i % 30 == 0:
                # 🔧 FIX: 不过滤 None，保持索引对齐
                pickle.dump({'data': data, 'text': text, 'wrong_indexes': wrong_indexes, 'wrong_reason': wrong_reason, 'results': results}, open(f"result2/{pickle_filename}", "wb"))
                print(f"Progress saved: {i}/{len(node_index_list)} nodes processed")
    
    else:
        # Concurrent processing using ThreadPoolExecutor
        print(f"Starting concurrent processing with {max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {executor.submit(process_single_node, i, node_index_list[i]): i 
                           for i in range(len(node_index_list))}
            
            # Process results as they complete
            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    node_result = future.result()
                    
                    if node_result is None:
                        # Node was skipped (no edges)
                        completed += 1
                        continue
                    
                    # Process the result
                    if node_result.get('explain', False):
                        explainations.append(node_result['result'])
                    else:
                        count += node_result['result']
                        # 🔧 FIX: 使用索引赋值而不是 append
                        wrong_reason[idx] = node_result['reason']
                        results[idx] = node_result['result2']
                        print(f"Completed index {idx}: Prediction = {node_result['result']}")
                        if node_result['result'] == 0:  # If the prediction is wrong
                            # 🔧 FIX: 存储全局节点索引而不是测试列表位置
                            wrong_indexes.append(node_index_list[idx])
                    
                    if node_result.get('failed', False):
                        count += (node_result['result2'] == text['label'][node_index_list[idx]])
                    
                    completed += 1
                    
                    if completed % 30 == 0:
                        # 🔧 FIX: 不过滤 None，保持索引对齐
                        pickle.dump({'data': data, 'text': text, 'wrong_indexes': wrong_indexes, 'wrong_reason': wrong_reason, 'results': results}, open(pickle_filename, "wb"))
                        print(f"Progress saved: {completed}/{len(node_index_list)} nodes completed")
                
                except Exception as exc:
                    print(f"Node at index {idx} generated an exception: {exc}")
                    completed += 1
        
        print(f"Concurrent processing completed: {completed}/{len(node_index_list)} nodes processed")
    
    # 🔧 FIX: 检查缺失的节点并输出
    print("\n" + "=" * 80)
    print("最终验证")
    print("=" * 80)
    
    missing_indices = []
    processed_count = 0
    for i in range(len(node_index_list)):
        if results[i] is None:
            missing_indices.append(i)
        else:
            processed_count += 1
    
    if missing_indices:
        print(f"\n⚠️  警告: {len(missing_indices)} 个节点处理失败或被跳过:")
        for i in missing_indices[:10]:  # 只显示前10个
            node_idx = node_index_list[i]
            print(f"  - 测试列表位置 {i}, 全局节点索引 {node_idx}")
        if len(missing_indices) > 10:
            print(f"  ... 还有 {len(missing_indices) - 10} 个节点")
    else:
        print("✅ 所有节点都已成功处理")
    
    # 🔧 FIX: 不过滤 None，保持索引对齐
    # results 和 wrong_reason 保留 None 值，确保 results[i] 对应 node_index_list[i]
    
    print(f"\n处理的节点总数: {processed_count}/{len(node_index_list)}")
    
    # 🔧 FIX: 重新验证准确率 (使用鲁棒判断标准)
    print("\n" + "-" * 80)
    print("重新计算准确率并验证 (使用鲁棒判断标准)")
    print("-" * 80)
    print("判断方法:")
    print("  1. 使用 get_matched_option 从预测中提取类别")
    print("  2. 如果提取成功，比较提取的类别与真实标签")
    print("  3. 如果提取失败，使用规范化比较作为后备")
    
    # 使用 results 数组重新计算准确率
    # 🔧 FIX: 使用鲁棒判断标准（与单个样本处理一致）
    recalculated_correct = 0
    recalculated_wrong_indexes = []
    comparison_methods = {'matched_option': 0, 'matched_option_normalized': 0, 'normalized': 0, 'exact': 0, 'mismatch': 0}
    
    for i in range(len(node_index_list)):
        if results[i] is not None:
            node_idx = node_index_list[i]
            predicted = results[i]
            true_label = text['label'][node_idx]
            
            is_correct = False
            method = 'mismatch'
            
            # 方法1: 精确匹配
            if predicted == true_label:
                is_correct = True
                method = 'exact'
            # 方法2: 使用 get_matched_option 提取预测类别
            elif options:
                extracted_pred = get_matched_option(predicted, options)
                if extracted_pred:
                    if extracted_pred == true_label:
                        is_correct = True
                        method = 'matched_option'
                    else:
                        # 尝试规范化比较提取的类别
                        norm_extracted = normalize_label(extracted_pred)
                        norm_true = normalize_label(true_label)
                        if norm_extracted == norm_true:
                            is_correct = True
                            method = 'matched_option_normalized'
            
            # 方法3: 规范化比较（后备）
            if not is_correct:
                normalized_pred = normalize_label(predicted)
                normalized_true = normalize_label(true_label)
                if normalized_pred == normalized_true:
                    is_correct = True
                    method = 'normalized'
            
            comparison_methods[method] += 1
            
            if is_correct:
                recalculated_correct += 1
            else:
                recalculated_wrong_indexes.append(node_idx)
    
    recalculated_accuracy = recalculated_correct / processed_count if processed_count > 0 else 0
    original_accuracy = count / processed_count if processed_count > 0 else 0
    
    print(f"\n原始统计:")
    print(f"  - 正确数量: {count}")
    print(f"  - 错误数量: {len(wrong_indexes)}")
    print(f"  - 准确率: {original_accuracy:.4f} ({original_accuracy*100:.2f}%)")
    
    print(f"\n重新验证统计 (使用鲁棒判断):")
    print(f"  - 正确数量: {recalculated_correct}")
    print(f"  - 错误数量: {len(recalculated_wrong_indexes)}")
    print(f"  - 准确率: {recalculated_accuracy:.4f} ({recalculated_accuracy*100:.2f}%)")
    
    print(f"\n比较方法统计:")
    for method, count_val in sorted(comparison_methods.items(), key=lambda x: x[1], reverse=True):
        if count_val > 0:
            print(f"  - {method}: {count_val} 个节点 ({count_val/processed_count*100:.1f}%)")
    
    # 检查一致性
    accuracy_diff = abs(original_accuracy - recalculated_accuracy)
    if accuracy_diff < 1e-6:
        print(f"\n✅ 验证通过: 准确率一致 (差异: {accuracy_diff:.10f})")
    else:
        print(f"\n⚠️  警告: 准确率不一致!")
        print(f"  - 差异: {accuracy_diff:.10f}")
        print(f"  - 原始错误索引数量: {len(wrong_indexes)}")
        print(f"  - 重新计算错误索引数量: {len(recalculated_wrong_indexes)}")
        
        # 检查错误索引的差异
        wrong_set1 = set(wrong_indexes)
        wrong_set2 = set(recalculated_wrong_indexes)
        
        only_in_original = wrong_set1 - wrong_set2
        only_in_recalculated = wrong_set2 - wrong_set1
        
        if only_in_original:
            print(f"\n  仅在原始错误索引中的节点 ({len(only_in_original)} 个):")
            for node_idx in list(only_in_original)[:5]:
                idx_in_list = node_index_list.index(node_idx) if node_idx in node_index_list else None
                if idx_in_list is not None:
                    pred = results[idx_in_list]
                    true = text['label'][node_idx]
                    print(f"    - 节点 {node_idx}: 预测='{pred}', 真实='{true}'")
                    print(f"      规范化后: 预测='{normalize_label(pred)}', 真实='{normalize_label(true)}'")
        
        if only_in_recalculated:
            print(f"\n  仅在重新计算错误索引中的节点 ({len(only_in_recalculated)} 个):")
            print(f"  (注意: 使用规范化标签比较，去除标点、空格、大小写差异)")
            for node_idx in list(only_in_recalculated)[:5]:
                idx_in_list = node_index_list.index(node_idx) if node_idx in node_index_list else None
                if idx_in_list is not None:
                    pred = results[idx_in_list]
                    true = text['label'][node_idx]
                    print(f"    - 节点 {node_idx}: 预测='{pred}', 真实='{true}'")
                    print(f"      规范化后: 预测='{normalize_label(pred)}', 真实='{normalize_label(true)}'")
    
    # 🔧 FIX: 最终保存（保留 None 值以保持索引对齐）
    final_data = {
        'data': data, 
        'text': text, 
        'wrong_indexes': recalculated_wrong_indexes,  # 使用验证后的错误索引（全局节点ID）
        'wrong_reason': wrong_reason,  # 保留 None，与 node_index_list 索引对齐
        'results': results,  # 保留 None，与 node_index_list 索引对齐
        'missing_indices': missing_indices,  # 处理失败的节点在测试列表中的位置
        'original_wrong_indexes': wrong_indexes,  # 保留原始错误索引用于调试
        'recalculated_accuracy': recalculated_accuracy,
        'original_accuracy': original_accuracy,
        'processed_count': processed_count,  # 实际处理的节点数量
        'total_count': len(node_index_list),  # 总节点数量
        'comparison_methods': comparison_methods  # 比较方法统计
    }
    pickle.dump(final_data, open(pickle_filename, "wb"))
    print(f"\n最终结果已保存到: {pickle_filename}")
    print(f"  - results 和 wrong_reason 保持与 node_index_list 索引对齐")
    print(f"  - results[i] 对应 node_index_list[i] (None 表示处理失败)")
    print(f"  - 使用鲁棒判断标准进行验证 (get_matched_option + 规范化比较)")
    
    print("=" * 80)
    
    if explain:
        return explainations
    return recalculated_accuracy, recalculated_wrong_indexes
