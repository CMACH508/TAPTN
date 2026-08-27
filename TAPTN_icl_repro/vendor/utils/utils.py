import random
import re
import torch
import numpy as np
import json
import os
import openai
from utils.load_arxiv import get_raw_text_arxiv
from utils.load_cora_year import get_raw_text_cora as get_raw_text_cora_year
from utils.load_cora import get_raw_text_cora
from utils.load_pubmed import get_raw_text_pubmed
from utils.load_arxiv_2023 import get_raw_text_arxiv_2023
from utils.load_tape_2023 import get_raw_text_tape_2023
from utils.load_products import get_raw_text_products
from utils.load_wisconsin import load_wisconsin
from time import sleep
from utils.prompts import generate_system_prompt, arxiv_natural_lang_mapping, refining3

from time import sleep
from random import randint
import threading
import json
import pickle
from utils.load_products import products_keys_list, products_mapping
from debate import Agent
import torch
import torch_geometric
torch.serialization.add_safe_globals([torch_geometric.data.data.DataEdgeAttr])


# 创建全局 Agent 实例用于 LLM 交互
# 注意：Agent 类内部已配置 API 密钥和 base_url，无需外部传入
global_agent = Agent(name="TAPTN Assistant", role="Graph Node Classification Expert")

#openai.api_key  = os.environ['OPENAI_API_KEY']
openai.api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("TAPTN_LLM_API_KEY") or ""
#neighbor_num = pickle.load(open(f"neighbor_num.pkl", "rb"))
def load_data(dataset, use_text=False, seed=0):
    """
    Load data based on the dataset name.

    Parameters:
        dataset (str): Name of the dataset to be loaded. Options are "cora", "pubmed", "arxiv", "arxiv_2023", and "product".
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
    elif dataset == "cora_year":
        data, data2, text = get_raw_text_cora_year(use_text, seed)
    elif dataset in ["wisconsin", "texas", "cornell", "washington"]:
        data, text = load_wisconsin(dataset)
    elif dataset == "pubmed":
        data, text = get_raw_text_pubmed(use_text, seed)
    elif dataset == "arxiv":
        data, text = get_raw_text_arxiv(use_text)
    elif dataset == "arxiv_2023":
        data, text = get_raw_text_arxiv_2023(use_text)
    elif dataset == "tape_2023":
        data, text = get_raw_text_tape_2023(use_text)
    elif dataset == "product":
        data, text = get_raw_text_products(use_text)
    else:
        raise ValueError("Dataset must be one of: cora, pubmed, arxiv")
    return data, data2, text

# from debate import Agent, F_sg
# agents = [
#         Agent("Adam", "a creative inventor who dares to break through conventional thinking. You approach problems by exploring multiple perspectives, challenging standard assumptions, and proposing innovative solutions. When analyzing input, think unconventionally, use analogies, and encourage out-of-the-box ideas. Your goal is to inspire creativity and novel approaches that push the boundaries of traditional problem-solving."),
#         Agent("Bob", "a rigorous and critically minded engineer with a keen eye for identifying and correcting logical and factual errors. Your focus is on dissecting arguments, verifying facts, and ensuring internal consistency. When presented with a problem, provide precise, well-reasoned critiques, and propose solutions that address any identified inaccuracies or flawed reasoning. Your goal is to ensure that every idea meets high standards of precision and reliability."),
#         Agent("David", "an experienced technical expert who considers problems rigorously and comprehensively. With a deep understanding of the development history and the cutting-edge technological frontiers in your field, you provide insights that integrate both historical perspective and current trends. Analyze challenges by grounding your responses in established principles while incorporating the latest advancements. Your goal is to deliver thorough, context-aware, and forward-thinking solutions.")
#     ]
def get_subgraph(node_idx, edge_index, hop=1):
    """
    Get subgraph around a specific node up to a certain hop.

    Parameters:
        node_idx (int): Index of the node.
        edge_index (torch.Tensor): Edge index tensor.
        hop (int, optional): Number of hops around the node to consider. Default is 1.

    Returns:
        list: Lists of nodes for each hop distance.
    """

    current_nodes = torch.tensor([node_idx])
    all_hops = []

    for _ in range(hop):
        mask = torch.isin(edge_index[0], current_nodes) | torch.isin(edge_index[1], current_nodes)
        
        # Add both the source and target nodes involved in the edges 
        new_nodes = torch.unique(torch.cat((edge_index[0][mask], edge_index[1][mask])))

        # Remove the current nodes to get only the new nodes added in this hop
        diff_nodes_set = set(new_nodes.numpy()) - set(current_nodes.numpy())
        diff_nodes = torch.tensor(list(diff_nodes_set))  
        
        all_hops.append(diff_nodes.tolist())

        # Update current nodes for the next iteration
        current_nodes = torch.unique(torch.cat((current_nodes, new_nodes)))

    return all_hops

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

def get_subgraph3(node_idx, edge_index, hop=1):
    """
    Get subgraph around a specific node up to a certain hop in a directed graph.
    Only composed edges of the same direction are allowed.

    Parameters:
        node_idx (int): Index of the node.
        edge_index (torch.Tensor): Edge index tensor.
        hop (int, optional): Number of hops around the node to consider. Default is 1.

    Returns:
        list: Lists of nodes for each hop distance, considering the direction of the edges.
    """

    current_nodes_src = torch.tensor([node_idx])
    current_nodes_dst = torch.tensor([node_idx])
    all_hops = []

    for _ in range(hop):
        mask_src = torch.isin(edge_index[0], current_nodes_src)
        mask_dst = torch.isin(edge_index[1], current_nodes_dst)

        # Add both the source and target nodes involved in the edges 
        new_nodes_src = torch.unique(edge_index[1][mask_src])
        new_nodes_dst = torch.unique(edge_index[0][mask_dst])

        # Remove the center node from the new nodes
        new_nodes_src = new_nodes_src[new_nodes_src != node_idx]
        new_nodes_dst = new_nodes_dst[new_nodes_dst != node_idx]

        # Store the nodes separately based on the direction of the edge
        all_hops.append(new_nodes_src.tolist())
        all_hops.append(new_nodes_dst.tolist())

        # Update the current nodes for the next iteration
        current_nodes_src = new_nodes_src
        current_nodes_dst = new_nodes_dst

    return all_hops

def sample_test_nodes(data, text, sample_size, dataset):
    """
    Randomly sample test nodes for evaluation.

    Parameters:
        data: Graph data object.
        text: Textual information associated with nodes.
        sample_size (int): Number of test nodes to sample.
        dataset (str): Name of the dataset being used.

    Returns:
        list: Indices of sampled test nodes.
    """

    np.random.seed(42)
    test_indices = np.where(data.test_mask.numpy())[0]

    if dataset != "product":
        sampled_indices = np.random.choice(test_indices, size=sample_size, replace=False)
        sampled_indices = sampled_indices.tolist()

    else:
        # Sample 2 times the sample size
        # node_indices = sample_test_nodes(data, 2 * sample_size)
        sampled_indices_double = np.random.choice(test_indices, size=2*sample_size, replace=False)

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

def count_tokens(messages):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    encoding = tiktoken.get_encoding("cl100k_base")

    total_tokens = 0
    for message in messages:
        total_tokens += len(encoding.encode(message['content']))

    return total_tokens

class MessageTooLongError(Exception):
        """Exception raised when the total number of tokens in the messages exceeds the limit of 16,384 tokens."""
        pass

def get_completion_from_messages_(messages, 
                                 model="gpt-3.5-turbo-0125", 
                                 temperature=0, max_tokens=500):
    """
    Get completion from the OpenAI API based on the given messages.

    Parameters:
        messages (list): Messages to be sent to the OpenAI API.
        model (str, optional): The name of the model to be used. Default is "gpt-3.5-turbo".
        temperature (float, optional): Sampling temperature. Default is 0.
        max_tokens (int, optional): Maximum number of tokens for the response. Default is 500.

    Returns:
        str: The content of the completion message.
    """
    max_tokens = min(3000, 16385-count_tokens(messages)-20)
    max_tokens = 3000
    _key = os.environ.get("OPENAI_API_KEY") or os.environ.get("TAPTN_LLM_API_KEY") or ""
    _base = os.environ.get("OPENAI_BASE_URL") or os.environ.get("TAPTN_LLM_BASE_URL") or "https://openrouter.ai/api/v1"
    if not _key:
        raise RuntimeError("No API key. Export OPENAI_API_KEY or TAPTN_LLM_API_KEY")
    client = openai.OpenAI(api_key=_key, base_url=_base)
    #print('client created')
    #model = 'meta-llama/llama-3.1-8b-instruct'
    if count_tokens(messages) > 16*1024:
        raise ValueError("The total number of tokens in the messages exceeds the limit of 16,384 tokens.")
        #model = 'meta-llama/llama-3.2-11b-vision-instruct'
        #return 'Genetic Algorithms'
    # elif count_tokens(messages) > 4000:
    #     model='gpt-3.5-turbo-0125'
    #model='gpt-3.5-turbo-1106'
    #model='gpt-3.5-turbo'
    # model = 'meta-llama/llama-3.1-8b-instruct'
    # model = 'meta-llama/llama-3.2-11b-vision-instruct'

    

    for i in range(3):
        try:
            if model=='gpt-3.5-turbo-instruct':
                response = client.completions.create(
                model=model,
                prompt=messages[1]['content'],
                temperature=temperature, 
                max_tokens=max_tokens, 
            )    
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature, 
                    max_tokens=max_tokens, 
                )
            if response.choices[0].finish_reason!="stop":
                raise RuntimeError("The completion did not finish.")
            return response.choices[0].message.content
        except Exception as e:
            if e is ValueError:
                raise MessageTooLongError("The total number of tokens in the messages exceeds the limit of 16,384 tokens.")
            
            print(f"Error: {e}")
            #model='gpt-3.5-turbo-0125'
            sleep(30)
    raise RuntimeError("The completion did not finish after 3 times.")

    # import re
    # try:
    #     messages[1]['content'] = re.escape(messages[1]['content'])
    #     response = client.chat.completions.create(
    #         model=model,
    #         messages=messages,
    #         temperature=temperature, 
    #         max_tokens=max_tokens, 
    #     )
    #     return response.choices[0].message.content


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
    # 注意：model 通过参数显式透传（而非赋值 global_agent.model），
    # 避免多线程并发下不同调用互相覆盖模型名导致的串模型问题
    try:
        response = global_agent.get_robust_completion(
            messages=messages,
            description="LLM API call",
            min_length=10,
            max_retries=10,
            temperature=temperature,
            model=model
        )
        return response
    except Exception as e:
        # 如果 Agent 方法失败，抛出更明确的错误
        print(f"Agent get_robust_completion failed: {e}")
        raise RuntimeError(f"Failed to get completion from Agent: {e}")

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



def get_important_neighbors(node_index, neighbors, text, dataset, max_papers_1=5, k=5):
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
    target_title = text['title'][node_index]

    Target_word = "Product" if dataset == "product" else "Paper"

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

    response = get_completion_from_messages([message])

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


def handle_important_neighbors(node_index, text, dataset, all_hops, data, abstract_len, include_label, max_papers_1):
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
    Target_word = "Product" if dataset == "product" else "Paper"
    k = 5
    attention_dir = f"attention/{dataset}/attention_{k}"
    filename = f"{attention_dir}/{node_index}.json"
    
    if os.path.exists(filename):
        with open(filename, "r") as f:
            important_neighbors = json.load(f)
    else:
        neighbors = list(set(all_hops[0]))
        important_neighbors = get_important_neighbors(node_index, neighbors, text, max_papers_1, k)
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
    Target_word = "Product" if dataset == "product" else "Paper"

    for h in range(0, hop):
        neighbors_at_hop = all_hops[h]
        neighbors_at_hop = np.array(neighbors_at_hop)
        neighbors_at_hop = np.unique(neighbors_at_hop)
        if h == 0:
            neighbors_at_hop = neighbors_at_hop[:max_papers_1]
        else:
            neighbors_at_hop = neighbors_at_hop[:max_papers_2]

        if len(neighbors_at_hop) > 0:
            
            if dataset != 'product':
                prompt_str_hop = f"It has following neighbor papers at hop {h+1}:\n"
            else:
                prompt_str_hop = f"It has following neighbor products purchased toghther at hop {h+1}:\n"

            #neighbors=[]
            for i, neighbor_idx in enumerate(neighbors_at_hop):

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
                              max_papers_2, abstract_len, include_label, dataset, abs_trunc=True, initial_judgement=None, refining=False, anonymize_edges=False, product_neighbor_reasoning=False):
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

    Returns:
        str: Prompt string.
    """
    if not (initial_judgement is None or initial_judgement[0] is None):
        initial_result=initial_judgement[1]
        initial_judgement2=initial_judgement[2]
        initial_result2=initial_judgement[3]
        initial_judgement=initial_judgement[0]
    else:
        initial_judgement=None

    #max_papers_1=max_papers_2=100
    # if dataset == 'product':
    #     return ""

    prompt_str = ""
    if dataset in ["cora", "cora_year","arxiv","arxiv_2023","tape_2023","pubmed"]:
        Target_word = "Paper"
    elif dataset == "wisconsin":
        Target_word = "Webpage"
    else:
        Target_word = "Frequently purchased-together item"

    for h in range(0, hop):
        if h == 0:
            if dataset=="product":
                neighbors_at_hop = all_hops[:1]
            else:
                neighbors_at_hop = all_hops[:2]
        elif h == 1:
            if dataset=="product":
                neighbors_at_hop = all_hops[1:2]    
            else:
                neighbors_at_hop = all_hops[2:6]
            #neighbors_at_hop = all_hops[2:6]
        else:
            neighbors_at_hop = all_hops[2*h+2:2*h+4]

        for i, direction in enumerate(neighbors_at_hop):
            if len(direction) == 0:
                continue
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

            if h == 0:
                if dataset in ["cora", "cora_year","arxiv","arxiv_2023","tape_2023","pubmed"]:
                    if refining:
                        prompt_str_head = "\n#### References ####\n\nThe paper has following references:\n" if i % 2 == 0 else "\n#### Citations ####\n\nThe paper has following citations:\n"
                    else:
                        prompt_str_head = "\nIt has following references:\n" if i % 2 == 0 else "\nIt has following citations:\n"
                elif dataset == "wisconsin":
                    prompt_str_head = "\nIt has outbound links to following webpages:\n" if i % 2 == 1 else "\nIt has inbound links from following webpages:\n"
                    #prompt_str_head = "\nIt has following outgoing hyperlinks:\n" if i % 2 == 0 else "\nIt is linked by following incoming hyperlinks:\n"
                    #prompt_str_head = "\nIt has outgoing hyperlinks to the following webpages:\n" if i % 2 == 0 else "\nIt is linked by incoming hyperlinks from the following webpages:\n"
                    # prompt_str_head = "\nIt has outgoing hyperlinks to the following webpages:\n" if i % 2 == 0 else "\nIt has incoming hyperlinks from the following webpages:\n"
                else:
                    prompt_str_head = "\nIt has following frequently purchased-together items:\n" if i % 2 == 0 else "\nIt is purchased together with following products:\n"

            if h==1 and dataset=="product":
                prompt_str_head = "\nIt has following frequently purchased-together items at hop 2:\n"
            #trunc reference list
            if h==0:
                # max_papers_1=min(max(neighbor_num[node_index]-len(neighbors_at_hop[1]),20),len(direction))
                # np.random.seed(42)
                # if len(direction)>max_papers_1:
                #     direction = np.random.choice(direction[:40], max_papers_1, replace=False)
                #     direction = np.sort(direction)
                # direction = np.random.choice(direction, max_papers_1, replace=False)
                # np.random.shuffle(direction)
                if dataset == "wisconsin":
                    np.random.seed(42)
                    if len(direction)>max_papers_1:
                        direction = np.random.choice(direction, max_papers_1, replace=False)
                else:
                    direction=direction[:max_papers_1]
            else:
                direction=direction[:max_papers_2]
            if dataset in ["cora", "cora_year","arxiv","arxiv_2023","tape_2023","pubmed"]:
                Target_word = "reference" if i % 2 == 0 else "citation"
            elif dataset == "wisconsin":
                Target_word = "outgoing hyperlink" if i % 2 == 0 else "incoming hyperlink"
                #Target_word = "links to webpage" if i % 2 == 0 else "linked from webpage"
                #Target_word = "It has outbound link to webpage OP" if i % 2 == 1 else "It has inbound link from webpage IP"
                Target_word = "It has outbound link to webpage " if i % 2 == 1 else "It has inbound link from webpage "

            # 匿名化边：将所有边方向标签统一替换为 "neighbor"
            if anonymize_edges:
                if h == 0:
                    prompt_str_head = "\nIt has following neighbors:\n"
                else:
                    prompt_str_head = f"\nIt has following neighbors at hop {h+1}:\n"
                if dataset == "wisconsin":
                    Target_word = "Neighbor page "
                elif dataset != "product":
                    Target_word = "Neighbor"

            prompt_str_body = ""
            
            outbound_neighbors_dict={}
            inbound_neighbors_dict={}
            if dataset == "wisconsin":
                for j, neighbor_idx in enumerate(direction):
                    neighbors = get_subgraph2(neighbor_idx, data.edge_index, hop=1)
                    outbound_neighbors = neighbors[0]
                    inbound_neighbors = neighbors[1]
                    outbound_neighbors_dict[neighbor_idx]=outbound_neighbors
                    inbound_neighbors_dict[neighbor_idx]=inbound_neighbors
                random_reconnect = False
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
            for j, neighbor_idx in enumerate(direction):
                prompt_str_body = ""
                
                if dataset in ["cora", "cora_year","arxiv","arxiv_2023","tape_2023","pubmed"]:
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

                elif dataset == "product":
                    if refining:
                        if abs_trunc:
                            if abstract_len > 0:
                                neighbor_abstract = text['content'][neighbor_idx]
                                prompt_str_body += f"## Frequently purchased-together item {j+1} product descriptions ##\n {neighbor_abstract[:abstract_len]}\n"
                        else:
                            neighbor_abstract = text['content'][neighbor_idx]
                            prompt_str_body += f"## Frequently purchased-together item {j+1} product descriptions ##\n {neighbor_abstract}\n"
                    else:
                        if abs_trunc:
                            if abstract_len > 0:
                                neighbor_abstract = text['content'][neighbor_idx]
                                prompt_str_body += f"Frequently purchased-together item {j+1} product descriptions: {neighbor_abstract[:abstract_len]}\n"
                        else:
                            neighbor_abstract = text['content'][neighbor_idx]
                            prompt_str_body += f"Frequently purchased-together item {j+1} product descriptions: {neighbor_abstract}\n"

                elif dataset == "wisconsin":
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
                            # neighbor_abstract = text['label'][neighbor_idx]
                            # if neighbor_abstract == "other":
                            #     neighbor_abstract += f", {text['abs'][neighbor_idx]}" 
                            # #prompt_str_body += f"{Target_word} {j+1} {abs_word}: {neighbor_abstract[:abstract_len]}\n"
                            # prompt_str_body += f"{Target_word}{text['title'][neighbor_idx]} which is a {text['label'][neighbor_idx]} page"+(f"with  content abstract as below: {text['abs'][neighbor_idx][:abstract_len]}\n" if text['label'][neighbor_idx]=='other' else '\n')
                            # prompt_str_body += ", with link patterns as below:\n"
                            # prompt_str_body += f"{neighbor_summary}\n"
                            prompt_str_body += f"{Target_word}{text['title'][neighbor_idx]} with content abstract as below: {text['abs'][neighbor_idx][:abstract_len]}\n"
                    else:
                        neighbor_abstract = text['abs'][neighbor_idx]
                        prompt_str_body += f"{Target_word} {j+1} {abs_word}: {neighbor_abstract}\n"

                neighbor_title = text['title'][neighbor_idx]
                
                if dataset in ["cora", "cora_year","arxiv","arxiv_2023","tape_2023","pubmed"]:
                    if refining:
                        prompt_str_body += f"## {Target_word} {j+1} title ##\n{neighbor_title}\n"
                    else:
                        prompt_str_body += f"{Target_word} {j+1} title: {neighbor_title}\n"

                elif dataset == "product":
                    if refining:
                        prompt_str_body += f"## Frequently purchased-together item {j+1} product name ##\n{neighbor_title}\n"
                    else:
                        prompt_str_body += f"Frequently purchased-together item {j+1} product name: {neighbor_title}\n"
                
                # elif dataset == "wisconsin":    
                #     prompt_str_body += f"{Target_word} {j+1} URL: {neighbor_title}\n"
                
                if initial_judgement is not None and neighbor_idx in initial_judgement:
                    neighbor_init = initial_judgement[neighbor_idx]
                    neighbor_init2 = initial_judgement2[neighbor_idx] if (initial_judgement2 is not None and neighbor_idx in initial_judgement2) else ""
                    neighbor_result = None
                    if initial_result is not None:
                        neighbor_result = initial_result.get(neighbor_idx)
                        # Remove 'number.' pattern (not applicable for product/wisconsin)
                        if neighbor_result is not None and dataset != "product" and dataset != "wisconsin":
                            neighbor_result = neighbor_result.replace('\n',',')
                            neighbor_result = re.sub(r'\d+\.\s*', '', neighbor_result)
                            neighbor_result2 = (initial_result2.get(neighbor_idx) or "").replace('\n','; ')
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
                    if neighbor_result is not None:
                        if dataset == "product" and not product_neighbor_reasoning:
                            # For product: show only categorization result (no full reasoning) to save tokens
                            prompt_str_body = f"## {Target_word} {j+1} initial categorization ##\n{neighbor_result}\n\n" + prompt_str_body
                        elif dataset != "wisconsin":
                            prompt_str_body = f"## {Target_word} {j+1} initial categorization ##\n{neighbor_result}\n## {Target_word} {j+1} reasons for initial categorization ##\n{neighbor_init}\n\n" + prompt_str_body
                        else:
                            prompt_str_body = f"## Initial categorization for {neighbor_title}: ##\n{neighbor_result}\n## Reasons of initial categorization for {neighbor_title}: ##\n{neighbor_init}\n\n" + prompt_str_body
                    else:
                        prompt_str_body = f"## {Target_word} {j+1} initial categorization and reasons ##\n{neighbor_init}\n\n" + prompt_str_body
                    #prompt_str_body = f"##{Target_word} {j+1} initial categorization and reasons##\n{neighbor_result2}\n{neighbor_init2}\n\n" + f"##{Target_word} {j+1} initial revision result and reasons##\n{neighbor_result2} -> {neighbor_result}\n{neighbor_init}\n\n" + prompt_str_body
                    #neighbor["initial judgement and reason"]=neighbor_init

                if include_label and (data.train_mask[neighbor_idx] or data.val_mask[neighbor_idx]):
                    label = text['label'][neighbor_idx]
                    prompt_str_body += f"Label: {label}\n"

                #prompt_str_body = f"{Target_word} {j+1} relation with the target paper: reference of the target paper\n"+prompt_str_body if i % 2 == 0 else f"{Target_word} {j+1} relation with the target paper: citation of the target paper\n"+prompt_str_body
                if dataset == "product":
                    prompt_str += f"\n### Frequently purchased-together item {j+1} ###\n\n"+prompt_str_body
                elif dataset!="wisconsin":
                    prompt_str += f"\n### {Target_word} {j+1} ###\n\n"+prompt_str_body
                else:
                    prompt_str += f"\n{prompt_str_body}"
                # if dataset == "product" and count_tokens([{'content':prompt_str}])>(16354-3000)*0.7:
                #     break

    return prompt_str

def get_node_info(node_indices, data, text, mode, dataset, source, hop=1, max_papers_1=20, max_papers_2=10, 
                  abstract_len=0, print_prompt=True, include_label=False, return_message=False, 
                  arxiv_style=False, include_options=False, include_abs=False, zero_shot_CoT=False, 
                  few_shot=False, use_attention=False, explain=False, initial_judgement=None, comfirm=False, options=None, revised_judgement=None, refining=False,
                  anonymize_edges=False, use_instructions=True, product_neighbor_reasoning=False):
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

    Returns:
        Depending on the 'return_message' flag, either prints the prompt and ideal answer or returns a list of messages.
    """
    if initial_judgement is not None:
        initial_result=initial_judgement[1]
        initial_result2=initial_judgement[3]
        initial_judgement2=initial_judgement[2]
        initial_judgement=initial_judgement[0]
    else:
        initial_judgement2=initial_result=initial_result2=None

    for node_index in node_indices:
        #info={}
        if mode == 'neighbors':
            # Initial setup for neighbors mode
            if dataset in ["cora", "cora_year","arxiv","arxiv_2023","tape_2023","pubmed"]:
                title = text['title'][node_index]
                if refining:
                    prompt_str = f"## Title ##\n{title}\n"
                else:
                    prompt_str = f"Title: {title}\n"
                #info['title']=title
                #prompt_str = "<Target Paper>\n"
            elif dataset == "wisconsin":
                title = text['title'][node_index]
                prompt_str = f"URL: {title}\n"
                #info['title']=title
                #prompt_str = "<Target Webpage>\n"
            elif dataset == "product":
                title = text['title'][node_index]
                prompt_str = f"Product Name: {title}\n"
                #info['title']=title
                #prompt_str = "<Target Product>\n"
            
            # Include abstract if required
            if include_abs:
                if source == 'product':
                    content = text['content'][node_index]
                    prompt_str = f"Product Description: {content}\n" + prompt_str
                elif source in ['cora', 'cora_year','arxiv','arxiv_2023',"tape_2023",'pubmed']:
                    abstract = text['abs'][node_index]
                    if refining:
                        prompt_str = f"## Abstract ##\n{abstract}\n" + prompt_str
                        # prompt_str += f"## References Number ##\n{data.out_degree[node_index]}\n"
                        # prompt_str += f"## Citations number ##\n{data.in_degree[node_index]}\n"
                    else:
                        prompt_str = f"Abstract: {abstract}\n" + prompt_str
                    prompt_str += f"References Number:\n{data.out_degree[node_index]}\n"
                    prompt_str += f"Citations number:\n{data.in_degree[node_index]}\n"   
                    #info['abstract']=abstract   
                    #prompt_str += f"\t<Abstract>{abstract}</Abstract>\n" 
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
                    
                    #prompt_str = f"{abs_word}:\n {abstract}\n" + prompt_str
                    prompt_str = f"##Outgoing links number ##\n{data.out_degree[node_index]}\n"+ prompt_str
                    prompt_str = f"##Incoming links number ##\n{data.in_degree[node_index]}\n"+ prompt_str   
                    # prompt_str = f"## Outbound Links Summary ##\n{outbound_summary}\n" + prompt_str
                    # prompt_str = f"## Inbound Links Summary ##\n{inbound_summary}\n" + prompt_str
                    #info['url']=url
                    #prompt_str += f"\t<URL>{url}</URL>\n"

            #prompt_str += f"\t<Title>{title}</Title>\n"
            if  comfirm:
                prompt_str += "Revised categorization and reasons: "+f"{revised_judgement}"+"\n" 
            if initial_judgement is not None:   
                target_init = initial_judgement[node_index]
                target_init2 = initial_judgement2[node_index]
                if initial_result is not None:
                    target_result = initial_result[node_index]
                    if dataset != "product" and dataset != "wisconsin": 
                        target_result = target_result.replace('\n',',')
                        target_result = re.sub(r'\d+\.\s*', '', target_result)
                        target_result2 = initial_result2[node_index].replace('\n','; ')
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
                #prompt_str += "Initial categorization and reasons:\n"+f"{initial_result[node_index]}\n{target_init}\n\n"
                # ir=[]
                # options=["Rule Learning", "Neural Networks", "Case Based", "Genetic Algorithms", "Computational Learning Theory", "Reinforcement Learning", "Probabilistic Methods"]
                # for option in options:
                #     if option in target_result:
                #         ir.append(option)
                # ir=';'.join(ir)
                #prompt_str+= f"## Initial categorization ##\n{ir}\n## Reasons for initial categorization ##\n{target_init}\n\n"
                #prompt_str += "## Initial categorization and reasons ##\n"+f"{initial_result[node_index]}\n{target_init}\n\n"
                if initial_result is not None:
                    prompt_str += f"## Initial categorization ##\n{target_result}\n## Reasons for initial categorization ##\n{target_init}\n\n"
                else:
                    prompt_str += "## Initial categorization and reasons ##\n"+f"{target_init}\n\n"
                #prompt_str += "##Initial categorization and reasons##\n"+f"{target_result2}\n{target_init2}\n\n"+"##Initial revision result and reasons##\n"+f"{target_result2} -> {target_result}\n{target_init}\n\n"
                #info['initial judgement and reason']=initial_judgement[node_index]
                # target_init = initial_judgement[node_index]
                # prompt_str += f"\t<Initial judgement and reason>{target_init}</initial judgement and reason>\n</Target Paper>\n"
            
            sys_prompt_str = generate_system_prompt(source, arxiv_style=arxiv_style, include_options=include_options, exlain=explain, comfirm=comfirm, options=options, use_instructions=use_instructions)
            #orig:all_hops = get_subgraph(node_index, data.edge_index, hop)
            all_hops = get_subgraph2(node_index, data.edge_index, 2 if hop>=2 else 1)
            
            # Check for test nodes
            if data.train_mask[node_index] or data.val_mask[node_index]:
                print('node indices should only contain test nodes!!')

            # Handle neighbors based on attention
            if use_attention:
                prompt_str += handle_important_neighbors(node_index, text, dataset, all_hops, data, abstract_len, include_label, max_papers_1)
            else:
                # orig:prompt_str += handle_standard_neighbors(node_index, text, all_hops, data, hop, max_papers_1, max_papers_2, 
                #                                         abstract_len, include_label, dataset)
                # prompt_str += handle_standard_neighbors_v2(node_index, text, all_hops, data, hop, max_papers_1, max_papers_2, 
                #                                         abstract_len, include_label, dataset)
                prompt_str += handle_standard_neighbors_v2(node_index, text, all_hops, data, hop, max_papers_1, max_papers_2, 
                                                         1000, include_label, dataset, initial_judgement=[initial_judgement,initial_result,initial_judgement2,initial_result2], refining=refining,
                                                         anonymize_edges=anonymize_edges, product_neighbor_reasoning=product_neighbor_reasoning)
                # info['neighbor papers'] = handle_standard_neighbors(node_index, text, all_hops, data, hop, max_papers_1*2, max_papers_2*4, 
                #                                          1000, include_label, dataset, initial_judgement=initial_judgement)
            if refining:
                if dataset in ["cora", "cora_year","arxiv","arxiv_2023","tape_2023","pubmed"]:
                    prompt_str = '#### Paper ####\n'+prompt_str
                elif dataset == "wisconsin":
                    prompt_str = '#### Webpage ####\n'+prompt_str
            if dataset == "product":
                prompt_str = '#### Target Product ####\n'+prompt_str
            
            #prompt_str = json.dumps(info)+'\n'
            # Finalize prompt for neighbors mode
            if explain:
                # prompt_str += "Please further revise your initial judgement of the most appropriate category for the paper. The references and citations of the paper should also be taken into consideration. If multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, explain how it is present in the text.\n\nAnswer: \n\n"
                prompt_str += "If multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, explain how it is present in the text.\n\nAnswer: \n\n"
            else:
                if comfirm:
                    prompt_str += "Please confirm the revised categorization or choose a more appropriate category for the paper.  The references and citations of the paper should also be taken into consideration. Make single choice, do not give any reasoning or logic for your answer.\nAnswer: \n\n"
                else:
                    if dataset in ["cora", "cora_year","arxiv","arxiv_2023","tape_2023","pubmed"]:
                        if dataset == 'cora':
                            options=["Rule Learning", "Neural Networks", "Case Based", "Genetic Algorithms", "Computational Learning Theory", "Reinforcement Learning", "Probabilistic Methods"]
                        elif dataset == 'cora_year':
                            options=['earlier than 1990', '1990-1992', '1993-1994', '1995-1995', '1996-1996', '1997-1999', 'later than 2000']
                        elif dataset == 'arxiv' or dataset == 'arxiv_2023':
                            options=set([f'{key} ({arxiv_natural_lang_mapping[key]})' for key in ['cs.GT','cs.MA','cs.RO','cs.NE','cs.IR','cs.SI','cs.CY']])
                        elif dataset == 'tape_2023':
                            options=set([f'{key} ({arxiv_natural_lang_mapping[key]})' for key in ['cs.RO','cs.CL','cs.AI','cs.LG']])
                        elif dataset == 'pubmed':
                            options=['Disease', 'Treatment', 'Symptoms', 'Diagnosis', 'Prevention', 'Risk Factors', 'Outcomes']
                        prompt_str += "####Question####:\nPredict the 2 most appropriate category for the paper. For each category you predict, give a relevance score from 0 and 1. Make double choices from the given list of categories:{}\n\nAnswer:\n\n".format('\n'.join(options))
                    #prompt_str += "####Question####:\nPredict the 2 most appropriate categories for the paper.\n\nAnswer:\n\n"
                    #prompt_str += "\n####Question####\nPredict the 2 most appropriate categories for the paper.\n"
                    #prompt_str+="\nAnswer:\n\n"
                    elif dataset=='cora_year':
                        prompt_str += "####Question####:\nPredict the 2 most probable publication time periods for the paper. For each answer you predict, give a confidence score between 0 and 1. Make double choices from the given list of time periods:{}\n\nAnswer:\n\n".format('\n'.join(['earlier than 1990', '1990-1992', '1993-1994', '1995-1995', '1996-1996', '1997-1999', 'later than 2000']))
                    elif dataset in ('wisconsin', 'texas', 'cornell', 'washington'):
                        # prompt_str += "####Question####:\nPredict the 2 most appropriate categories for the webpage with URL: {}. For each category you predict, give a relevance score from 0 and 1. Make double choices from the given list of categories:{}\n\nNOTE: THE 5 STEPS IN THE STEP-BY-STEP CLASSIFICATION PROCESS MUST BE ALL FINISHED! NO DIRECT ANSWERING BEFORE THE END OF STEP-BY-STEP CLASSIFICATION PROCESS!\n\nAnswer:\n\n".format(text['title'][node_index],'\n'.join(['faculty', 'staff', 'department', 'course', 'project', 'student','other']))
                        prompt_str += "####Question####:\nPredict the 2 most appropriate categories for the webpage with URL: {}. For each category you predict, give a relevance score from 0 and 1. Make double choices from the given list of categories:{}\n\nAnswer:\n\n".format(text['title'][node_index],'\n'.join(['faculty', 'staff', 'department', 'course', 'project', 'student','other']))
                    
                    elif dataset=='product':
                        prompt_str += "####Question####:\nPredict the 2 most appropriate categories for the target product from Amazon. For each category you predict, give a relevance score from 0 and 1. Make double choices from the given list of categories:\n[\n{}\n]\nAnswer:\n\n".format(',\n'.join(products_keys_list))
                        # prompt_str += "####Question####:\nPredict the 2 most appropriate categories for the target product from Amazon. For each category you predict, give a relevance score from 0 and 1.\nAnswer:\n\n"
                        
                if zero_shot_CoT:
                    if dataset == 'cora':
                        prompt_str += "Let's think step by step.\n\n"
                        #prompt_str += "Let's think step by step.\n\nAnswer: \n\n"
                    else:
                        prompt_str += "Let's think step by step.\n\n"
                    
#                     prompt_str+="""1. **Understand the Categories**:\n\n
# - **Case Based**: Refers to case based reasoning, a method in artificial intelligence where new problems are solved by adapting solutions from similar past problems.

# - **Genetic Algorithms**: Represents papers on genetic algorithms, a class of optimization algorithms inspired by the process of natural selection in biology.

# - **Neural Networks**: Covers papers on neural networks, a set of algorithms modeled after the human brain, widely used in machine learning for tasks like image and speech recognition.

# - **Probabilistic Methods**: Encompasses research on probabilistic approaches in machine learning, including Bayesian networks and other methods that involve probability theory.

# - **Reinforcement Learning**: Includes papers on reinforcement learning, a type of machine learning where agents learn to make decisions by receiving rewards or penalties.

# - **Rule Learning**: Refers to methods focused on learning interpretable rules from data, often used in fields like data mining and knowledge discovery.

# - **Theory**: Represents theoretical research in machine learning, including the development of new algorithms and the mathematical foundations of machine learning.\n\n2. **Analyze the Paper's Abstract and Title**:\n\n- **Keywords and Phrases**:\n\n"""
#                     prompt_str+="""### Iterative Refinement Process ###

# ## 1. Reevaluate Information ##
# - **Deep Dive into Content**: Re-read abstracts, titles, categories refined in last iteration and reasons for the paper, references, and citations to uncover overlooked aspects.
# - **Focus on Keywords and Themes**: Identify specific keywords, technical terms, and prevalent themes that may have been undervalued.

# ## 2. Cross-reference Categories ##
# - **Comparison with Known Categories**: Compare identified themes and topics against definitions or typical contents of categories not previously considered.
# - **Category Definitions**: Revisit and clarify the definitions of the categories to ensure accurate alignment.

# ## 3. Analytical Redirection ##
# - **Hypothesis Testing**: Formulate hypotheses about possible correct categories based on new insights and test them by comparing the content of references and citations.
# - **Pattern Identification**: Identify recurring themes or methods in the paper and its network that align with a different category.

# ## 4. Define Refinement Criteria ##
# - **Category Connotation Clarification**: Clearly define and document the connotation and boundaries of each category. All the categories in the given option list should be considered.

# ## 5. Conduct Refinement ##
# - **Re-evaluate Initial Judgments**: Reassess the initial category judgments using clarified connotations and metrics.
# - **Re-assess References and Citations**: Review the category assignments of references and citations, looking for emerging patterns.
# - **Cross-check with Category Connotation**: Ensure assigned categories align with the refined understanding of each category.

# ## 6. Feedback Mechanism ##
# - **Mock Peer Review**: Simulate a peer review by re-evaluating with a fresh perspective.
# - **Iterative Review**: Introduce a process for periodic review to consider new understandings or approaches in categorization.

# ## 7. Final Adjustment and Check ##
# Make the final adjustments to the category based on the accumulated insights and feedback. After final adjustment, review the reasoning process and final categorization carefully and identify any factual errors, inconsistencies, or missing important information. If you find any issue, please fix it accordingly to ensure it logically fits with its content and its scholarly context. This final check ensures that the category reflects the paper's contributions and themes accurately.

# By combining detailed analysis, consistency checks, and thematic alignment with iterative refinement, you can enhance the accuracy and relevance of category assignments for each paper. This comprehensive approach ensures that categorization is well-informed by both the content of the paper and its academic context.

# ### End of The Refinement Process ###

# Now, follow the process above to refine the categorization of the paper.

# ### Iterative Refinement Process ###

# ## 1. Reevaluate Information ##\n\n"""
                
            # Return the message
            if return_message:
                #return [{'role':'system', 'content': sys_prompt_str}, {'role':'user', 'content': f"{sys_prompt_str}\n\n{prompt_str}"}]
                return [{'role':'system', 'content': sys_prompt_str}, {'role':'user', 'content': f"{prompt_str}"}]
        
        elif mode == 'ego':
            # Formulate the prompt
            sys_prompt_str_abs  = generate_system_prompt(source, arxiv_style, include_options=include_options, use_instructions=use_instructions)
            
            title = text['title'][node_index]

            few_shot_examples = ""
            if few_shot:
                with open(f"few_shot_examples/{dataset}.txt", 'r') as f:
                    few_shot_examples = f.read()

            # Check if the source is a product
            if source == 'product':
                content = text['content'][node_index]
                if include_abs:
                    prompt_str = f"{few_shot_examples}\nContent: {content}\nTitle: {title}\n"
                else:
                    prompt_str = f"{few_shot_examples}\nTitle: {title}\n"
            elif source == 'wisconsin':
                url = text['title'][node_index]
                content = text['abs'][node_index]
                if include_abs:
                    prompt_str = f"{few_shot_examples}\nContent Abstract: {content}\nURL: {url}\n"
                else:
                    prompt_str = f"{few_shot_examples}\nTitle: {url}\n"
            else:
                abstract = text['abs'][node_index]
                if include_abs:
                    prompt_str = f"{few_shot_examples}\nAbstract: {abstract}\nTitle: {title}\n"
                else:
                    prompt_str = f"{few_shot_examples}\nTitle: {title}\n"
            if initial_judgement is not None:   
                target_init = initial_judgement[node_index]
                prompt_str += "Initial judgement and reason: "+f"{target_init}"+"\n"
                #info['initial judgement and reason']=initial_judgement[node_index]
                # target_init = initial_judgement[node_index]
                # prompt_str += f"\t<Initial judgement and reason>{target_init}</initial judgement and reason>\n</Target Paper>\n"
            if zero_shot_CoT:
                prompt_str += "Answer: \n\n Let's think step by step.\n"
            else:
                if explain:
                    prompt_str += "Please further revise your initial judgement of the most appropriate category for the paper. If multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, explain how it is present in the text.\n\nAnswer: \n\n"
                else:
                    prompt_str += "Please further revise your initial judgement of the most appropriate category for the paper. Do not provide your reasoning.\nAnswer: \n\n"

            if return_message:
                return [{'role':'system', 
                        'content': sys_prompt_str_abs},    
                        {'role':'user', 
                        'content': f"{prompt_str}"}] 
            
        else:
            print('Invalid mode! Please use either "neighbors" or "abstract"')


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

def get_matched_option3(prediction, valid_options):
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
    previous_option = ""
    latest_position = 0
    previous_position = 0

    # Iteratively check each substring of the prediction
    for option in valid_options:
        position = prediction.rfind(option.lower())
        if position != -1 and position > latest_position:
            previous_option = matched_option
            matched_option = option
            previous_position = latest_position
            latest_position = position
    if previous_option == "":
        previous_option = matched_option
    if matched_option == "Computational Learning Theory":
        matched_option = "Theory"
    # Return the first matched option if available, else return an empty string
    return previous_option

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
                                           few_shot=False, use_attention=False, options=None, explain=False, initial_judgement=None,
                                           anonymize_edges=False, use_instructions=True,
                                           model_name='meta-llama/llama-3.3-70b-instruct',
                                           product_neighbor_reasoning=False):
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

    # WebKB 四校数据结构与提示词模板完全一致：texas/cornell/washington 在所有
    # 分支逻辑中按 'wisconsin' 处理（保存文件名由上层 dataset_name 决定，不受影响）
    if dataset in ('texas', 'cornell', 'washington'):
        dataset = 'wisconsin'

    refining=False
    message = get_node_info([node_index], data, text, hop=hop, dataset=dataset, source=source,
                            mode=mode, max_papers_1=max_papers_1, max_papers_2=max_papers_2, return_message=True, 
                            include_label=include_label, abstract_len=abstract_len, print_prompt=print_prompt,
                            arxiv_style=arxiv_style, include_options=include_options, 
                            zero_shot_CoT=zero_shot_CoT, few_shot=few_shot, include_abs=include_abs,
                            use_attention=use_attention, explain=explain, initial_judgement=initial_judgement, refining=refining,
                            anonymize_edges=anonymize_edges, use_instructions=use_instructions,
                            product_neighbor_reasoning=product_neighbor_reasoning)

    if print_out:
        print(message[0]['content'], end="\n\n")
        print(message[1]['content'], end="\n\n")

    ideal_answer = text['label'][node_index]
    
    print("Ideal_answer:", ideal_answer, end="\n\n")
    
    #return 1, message[1]['content'], ""
    # Get completion message and print
    response = get_completion_from_messages(message, model=model_name)
    if print_out:
        print(response)
    
    if source == "arxiv" and arxiv_style == "identifier": 
        response = response.lower()

    prediction = response if response is not None else ""
    print("Prediction: ", prediction, end="\n\n")
    if explain:
        return prediction
    # narrowed_options = list(set(get_matched_option2(initial_judgement[node_index], options)+get_matched_option2(prediction, options)))
    # message = get_node_info([node_index], data, text, hop=hop, dataset=dataset, source=source,
    #                         mode=mode, max_papers_1=max_papers_1, max_papers_2=max_papers_2, return_message=True, 
    #                         include_label=include_label, abstract_len=abstract_len, print_prompt=print_prompt,
    #                         arxiv_style=arxiv_style, include_options=include_options, 
    #                         zero_shot_CoT=zero_shot_CoT, few_shot=few_shot, include_abs=include_abs,
    #                         use_attention=use_attention, explain=explain, initial_judgement=initial_judgement, comfirm=True, options=narrowed_options, revised_judgement=get_matched_option2(prediction, options)[0] if get_matched_option2(prediction, options) else get_matched_option2(initial_judgement[node_index], options)[0])

    # if print_out:
    #     print(message[0]['content'], end="\n\n")
    #     print(message[1]['content'], end="\n\n")

    # ideal_answer = text['label'][node_index]
    
    # print("Ideal_answer:", ideal_answer, end="\n\n")
    
    # # Get completion message and print
    # response = get_completion_from_messages(message)
    # if print_out:
    #     print(response)
    
    # if source == "arxiv" and arxiv_style == "identifier": 
    #     response = response.lower()

    # prediction = response if response is not None else ""
    # print("Prediction: ", prediction, end="\n\n")
    # Compare the prediction with ideal_answer
    options_list='\n'.join(options)
    if dataset == 'arxiv' or dataset == 'arxiv_2023' or dataset == 'tape_2023':
        options_list='\n'.join([f'{key} ({arxiv_natural_lang_mapping[key]})' for key in options])
        if initial_judgement is None:
            message2=[{'role':'system', 'content': f'Please extract the refined most appropriate category for the paper. Choose from the following categories:\n\n{options_list}'}, {'role':'user', 'content': f"{prediction}\n\nAnswer: \n\n"}] 
        else:    
            message2=[{'role':'system', 'content': f'Please extract the most appropriate category for the paper. Choose from the following categories:\n\n{options_list}'}, {'role':'user', 'content': f"#### Initial Category and Reasons ####\n\n{initial_judgement[0][node_index]}\n\n#### Refined Category and Reasons #####\n\n{prediction}\n\nAnswer: \n\n"}] 
    if dataset == 'cora':
        message2=[{'role':'system', 'content': f'Please extract the most appropriate category for the paper. Make single choise from the following categories:\n\n{options_list}'}, {'role':'user', 'content': f"{prediction}\n\nAnswer: \n\n"}] 
    elif dataset=='cora_year':
        message2=[{'role':'system', 'content': f'Please extract the 2 most probable publication time periods for the paper. Make choises from the following time periods:\n\n{options_list}'}, {'role':'user', 'content': f"{prediction}\n\nAnswer: \n\n"}]
    elif dataset in ('wisconsin', 'texas', 'cornell', 'washington'):
        options_list='\n'.join(['faculty', 'staff', 'department', 'course', 'project', 'student'])
        message2=[{'role':'system', 'content': f'Please extract the category with highest relevance score for the webpage. Make single choise from the following categories:\n\n{options_list}'}, {'role':'user', 'content': f"{prediction}\n\nAnswer: \n\n"}]
    elif dataset=='product':
        options_list=',\n'.join(products_keys_list)
        message2=[{'role':'system', 'content': f'Please extract the most relevant category as well as its relevance score for the product. Make single choice from the following categories:\n[\n{options_list}\n]\nDirectly output your answer, don\'t give any explanation'}, {'role':'user', 'content': f"Please extract the most relevant category for the product based on the reasoning process. Make single choice from the following categories:\n[\n{options_list}\n]\n#### Reasoning Process ####\n\n{prediction}\n\nSo, the most relevant category of the product and its relevance score is: \n\n"}]
        message2=[{'role':'system', 'content': f'Please extract the category with highest relevance score as well as its relevance score for the product. Make single choice from the following categories:\n[\n{options_list}\n]\nDirectly output your answer, don\'t give any explanation.'}, {'role':'user', 'content': f"Please extract the most relevant category for the product based on the reasoning process. Make single choice from the following categories:\n[\n{options_list}\n]\n#### Reasoning Process ####\n\n{prediction}\n\nSo, the category with highest relevance score of the product and its relevance score is: \n\n"}]
        # message2=[{'role':'system', 'content': f'Please extract the two most relevant categories as well as the corresponding relevance scores for the product. Make double choices from the following categories:\n[\n{options_list}\n]\nThe extracted categories must be listed from most relevant to least relevant.'}, {'role':'user', 'content': f"#### Reasoning Process ####\n\n{prediction}\n\n#### Question ####\n\nPlease extract the two most relevant categories as well as the corresponding relevance scores for the product based on the reasoning process. Make double choices from the following categories:\n[\n{options_list}\n]\n\nAnswer: \n\n"}]
        # message2=[{'role':'system', 'content': f'Please extract the two categories with highest relevance scores as well as the corresponding relevance scores for the product. Make double choices from the following categories:\n[\n{options_list}\n]\nThe extracted categories must be listed from most relevant to least relevant.'}, {'role':'user', 'content': f"#### Reasoning Process ####\n\n{prediction}\n\n#### Question ####\n\nPlease extract the two categories with highest relevance scores as well as the corresponding relevance scores for the product based on the reasoning process. Make double choices from the following categories:\n[\n{options_list}\n]\n\nAnswer: \n\n"}]
    print(message2[0]['content'], end="\n\n")
    print(message2[1]['content'], end="\n\n")
    #prediction2 = get_completion_from_messages(message2,model='gpt-3.5-turbo-0125')
    prediction2 = get_completion_from_messages(message2,model='meta-llama/llama-3.1-8b-instruct')
    # prediction2 = get_completion_from_messages(message2,model='meta-llama/llama-3.2-11b-vision-instruct')
    # if zero_shot_CoT:
    #     # Use the helper function to get the last matched option
    #     if options == None:
    #         raise "options is not define!"
    #     prediction = get_matched_option(prediction, options)
    # if ',' in prediction:
    #     prediction = prediction.split(',')[0].strip()

    if prediction2 is not None:
        print("Prediction: ", prediction2)
        if dataset == 'product':
            prediction3 = get_matched_option(prediction2, options)
            if prediction3 == 'Home & Kitchen':
                prediction2 = get_completion_from_messages(message2,model='meta-llama/llama-3.1-8b-instruct')
                #prediction2 = get_completion_from_messages(message2,model='meta-llama/llama-3.2-11b-vision-instruct')
                prediction3 = get_matched_option(prediction2, options)     
        else: 
            prediction3 = get_matched_option(prediction2, options)   
        if prediction3 == "":
            message3=[{'role':'system', 'content': f'Please find the category that is most similar to the given category in the following list of categories:\n\n{options_list}'}, {'role':'user', 'content': f"{prediction2}\n\nAnswer: \n\n"}] 
            #prediction3 = get_completion_from_messages(message3,model='gpt-3.5-turbo-0125')
            if dataset == 'product':
                trial_count=0
                while prediction3 == "":
                    trial_count+=1
                    prediction3 = get_completion_from_messages(message2,model='meta-llama/llama-3.1-8b-instruct')
                    #prediction3 = get_completion_from_messages(message2,model='meta-llama/llama-3.2-11b-vision-instruct')
                    print("Prediction: ", prediction3)
                    prediction3 = get_matched_option(prediction3, options)
                    if trial_count>5:
                        prediction3 = get_completion_from_messages(message3,model='meta-llama/llama-3.1-8b-instruct')
                        #prediction2 = get_completion_from_messages(message3,model='meta-llama/llama-3.2-11b-vision-instruct')
                        print("Prediction: ", prediction3)
                        prediction3 = get_matched_option(prediction3, options)
                        break
                #prediction2 = prediction3
            else:
                prediction3 = get_completion_from_messages(message3,model='meta-llama/llama-3.1-8b-instruct')
                print("Prediction: ", prediction3)
                prediction3 = get_matched_option(prediction3, options)
        # Compare the prediction with ideal_answer
        if dataset=='cora_year':
            print("Is prediction correct? ", ideal_answer in prediction2.lower(), end="\n\n")
            return int(ideal_answer in prediction2.lower()), prediction, prediction2
        elif dataset=="product":
            print("Is prediction correct? ", products_mapping[prediction3] == ideal_answer, end="\n\n")
            
            return int(products_mapping[prediction3] == ideal_answer), prediction, prediction2
        else:
            print("Is prediction correct? ", prediction3 == ideal_answer, end="\n\n")
            
            return int(prediction3 == ideal_answer), prediction, prediction2   
    else:
        print("No valid prediction could be made.")


def process_and_compare_predictions(node_index_list, data, text, dataset_name, source, hop=2, 
                                    max_papers_1=20, max_papers_2=10, mode="title", 
                                    include_label=True, abstract_len=0, arxiv_style=False, 
                                    include_options=False, include_abs=False, zero_shot_CoT=False, 
                                    few_shot=False, use_attention=False, options=None, timeout=6000, explain=False, initial_judgement=None,
                                    max_workers=100, anonymize_edges=False, use_instructions=True,
                                    model_name='meta-llama/llama-3.3-70b-instruct',
                                    product_neighbor_reasoning=False):
    """
    Process and compare predictions for a list of node indices (concurrent version).

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
        timeout (int, optional): Per-node timeout in seconds. Default is 6000.
        max_workers (int, optional): Number of concurrent threads. Default is 10.

    Returns:
        tuple: The first element is the accuracy of the predictions, and the second is a list of wrong indexes.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    count = 0
    wrong_indexes = []
    wrong_reason = []
    results = []
    too_long_indexes = []
    explainations = []
    base_sleep_time = 0.5
    max_sleep_time = 60

    # 外层每个节点的总超时 = 单次超时 × (最大重试次数 + 1)，留足余量
    max_retries = 5
    outer_node_timeout = timeout * (max_retries + 1)

    def process_node(i):
        """处理单个节点，内部保留超时控制与重试机制，返回 (i, result, reason, result2, status)。

        防卡死设计：
          - 内层线程设为 daemon=True：t.join(timeout) 超时返回后，旧线程后台静默运行，
            不阻塞主流程，也不堆积为僵尸线程（进程退出时由 OS 自动回收）。
          - inner() 通过参数默认值显式绑定 result_holder/exception_holder，
            避免 Python 闭包延迟绑定在循环中引用错误变量。
          - 外层由 future.result(timeout=outer_node_timeout) 兜底，
            防止整个 process_node 函数（含所有重试+sleep）意外挂死。
        """
        node_index = node_index_list[i]

        # 跳过孤立节点
        if data.in_degree[node_index] == 0 and data.out_degree[node_index] == 0:
            return i, None, None, None, "isolated"

        for retry in range(max_retries):
            result_holder = [None]
            exception_holder = [None]

            # 用默认参数显式绑定本次循环的持有容器，避免闭包变量被下次循环覆盖
            def inner(_rh=result_holder, _eh=exception_holder, _retry=retry):
                try:
                    print(f"Processing index {i}, attempt {_retry + 1}...")
                    res, reason, res2 = print_node_info_and_compare_prediction(
                        node_index, data, text, dataset=dataset_name, source=source,
                        hop=hop, max_papers_1=max_papers_1,
                        max_papers_2=max_papers_2, mode=mode,
                        include_label=include_label, print_out=True,
                        arxiv_style=arxiv_style, include_options=include_options,
                        zero_shot_CoT=zero_shot_CoT, few_shot=few_shot, include_abs=include_abs,
                        use_attention=use_attention, options=options, explain=explain,
                        initial_judgement=initial_judgement,
                        anonymize_edges=anonymize_edges, use_instructions=use_instructions,
                        model_name=model_name,
                        product_neighbor_reasoning=product_neighbor_reasoning
                    )
                    _rh[0] = (res, reason, res2)
                except Exception as e:
                    _eh[0] = e

            # daemon=True：join 超时后不再等待，线程后台运行直至自然结束或进程退出
            t = threading.Thread(target=inner, daemon=True)
            t.start()
            t.join(timeout=timeout)
            # join 返回后，无论线程是否仍在运行，均继续向下处理

            if isinstance(exception_holder[0], MessageTooLongError):
                too_long_indexes.append(i)
                print(f"Message too long at index {i}, skip retries → fallback")
                break

            if result_holder[0] is not None:
                res, reason, res2 = result_holder[0]
                return i, res, reason, res2, None

            # 超时或其他异常 → 准备重试
            if exception_holder[0]:
                print(f"Error at index {i}, attempt {retry + 1}: {exception_holder[0]}")
            else:
                print(f"Timeout at index {i}, attempt {retry + 1} "
                      f"(daemon thread still running in background)")

            if retry < max_retries - 1:
                sleep_time = min(base_sleep_time * (2 ** retry) + randint(0, 1000) / 1000, max_sleep_time)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                sleep(sleep_time)

        # 所有重试耗尽 → fallback
        print(f"All retries exhausted for index {i}, using fallback prediction")
        if initial_judgement is not None and initial_judgement[1] is not None:
            fallback_pred = get_matched_option(initial_judgement[1][node_index], options)
            is_correct = int(fallback_pred == text['label'][node_index])
            fallback_reason = initial_judgement[0][node_index]
            print(f"Fallback prediction: {fallback_pred}")
            return i, is_correct, fallback_reason, fallback_pred, "fallback"
        return i, 0, "Failed after max retries", "", "failed"

    # ── 并发执行 ──────────────────────────────────────────────────────────────
    temp_results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(process_node, i): i for i in range(len(node_index_list))}

        for future in as_completed(future_to_idx):
            orig_i = future_to_idx[future]
            try:
                # 外层超时兜底：防止 process_node 整体（含所有重试+sleep）意外挂死
                i_idx, result, reason, result2, status = future.result(timeout=outer_node_timeout)
                temp_results[i_idx] = (result, reason, result2, status)
            except TimeoutError:
                print(f"[Outer timeout] Task index {orig_i} exceeded {outer_node_timeout}s, "
                      f"forcing fallback")
                node_index = node_index_list[orig_i]
                if initial_judgement is not None and initial_judgement[1] is not None:
                    fallback_pred = get_matched_option(initial_judgement[1][node_index], options)
                    is_correct = int(fallback_pred == text['label'][node_index])
                    temp_results[orig_i] = (is_correct, initial_judgement[0][node_index],
                                            fallback_pred, "outer_timeout_fallback")
                else:
                    temp_results[orig_i] = (0, "Outer timeout", "", "outer_timeout")
            except Exception as e:
                print(f"Unexpected error for task index {orig_i}: {e}")
                temp_results[orig_i] = (0, str(e), "", "error")

    # ── 按原始顺序聚合结果 ────────────────────────────────────────────────────
    processed_node_ids = []  # 与 results/wrong_reason 一一对应的真实节点 id（跳过孤立节点后）
    for i in range(len(node_index_list)):
        if i not in temp_results:
            continue
        result, reason, result2, status = temp_results[i]
        if status == "isolated":
            continue

        if explain:
            explainations.append(result)
        else:
            count += result
            wrong_reason.append(reason)
            results.append(result2)
            processed_node_ids.append(node_index_list[i])
            print(f"Index {i} - Correct: {result}")
            if result == 0:
                wrong_indexes.append(i)

    # ── 构造动态文件名 ────────────────────────────────────────────────────────
    # 格式: {dataset_name}_hop{hop}_{anon}_{instr}_{model}.pkl
    _anon_tag  = "anon"   if anonymize_edges  else "noanon"
    _instr_tag = "guide"  if use_instructions else "noguide"
    _model_tag = model_name.split("/")[-1]          # 取 "/" 后的最后一段
    _model_tag = _model_tag.replace(":", "_")       # 防止特殊字符出现在文件名中
    _base_name = f"{dataset_name}_hop{hop}_{_anon_tag}_{_instr_tag}_{_model_tag}"

    if explain:
        result_out = [None] * data.test_mask.shape[0]
        for i, node_index in enumerate(node_index_list):
            result_out[node_index] = explainations[i]
        _save_path = f"{_base_name}_explain.pkl"
        pickle.dump(result_out, open(_save_path, "wb"))
        print(f"Results saved to {_save_path}")
    else:
        accuracy = count / len(node_index_list)
        print("Accuracy:", accuracy)
        print("Wrong indexes:", wrong_indexes)
        print("Wrong indexes length:", len(wrong_indexes))

        # 按真实节点 id 建立索引字典（孤立节点被跳过时，键必须取自 processed_node_ids，
        # 否则自首个孤立节点位置起所有键整体前移错位）
        save_result  = {processed_node_ids[i]: results[i]      for i in range(len(results))}
        save_reason  = {processed_node_ids[i]: wrong_reason[i] for i in range(len(wrong_reason))}
        save_data = {
            "result":       save_result,
            "reason":       save_reason,
            "accuracy":     accuracy,
            "wrong_indexes": wrong_indexes,
        }
        _save_path = f"{_base_name}.pkl"
        pickle.dump(save_data, open(_save_path, "wb"))
        print(f"Results saved to {_save_path}")

        # 孤立节点被跳过时 wrong_indexes/count 仅覆盖已处理节点，不再用硬断言
        _n_skipped = len(node_index_list) - len(results)
        if _n_skipped > 0:
            print(f"[WARN] {_n_skipped} isolated/skipped nodes excluded from results "
                  f"(processed={len(results)}, accuracy denominator={len(node_index_list)})")
        return accuracy, wrong_indexes

def process_and_compare_predictions_cc(node_index_list, data, text, dataset_name, source, hop=2, 
                                    max_papers_1=20, max_papers_2=10, mode="title", 
                                    include_label=True, abstract_len=0, arxiv_style=False, 
                                    include_options=False, include_abs=False, zero_shot_CoT=False, 
                                    few_shot=False, use_attention=False, options=None, timeout=6000, explain=False, initial_judgement=None):
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

    Returns:
        tuple: The first element is the accuracy of the predictions, and the second is a list of wrong indexes.
    """
 
    i = 0
    count = 0
    wrong_indexes = []
    wrong_reason=[]
    results=[]
    too_long_indexes = []
    base_sleep_time = 0.5  # Starting sleep time
    max_sleep_time = 60  # Maximum sleep time
    explainations=[]

    # Dictionaries to store results and exceptions per thread
    result_containers = {}
    exception_containers = {}

    import threading
    import json
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Function to process each node index
    def process_node(i, retry_limit=5):
        retry_count = 0
        success = False
        while retry_count < retry_limit and not success:
            try:
                print(f"Processing index {i}, attempt {retry_count + 1}...")
                node_index = node_index_list[i]
                result, reason, result2 = print_node_info_and_compare_prediction(
                    node_index, data, text, dataset=dataset_name, source=source,
                    hop=hop, max_papers_1=max_papers_1, max_papers_2=max_papers_2,
                    mode=mode, include_label=include_label, print_out=True,
                    arxiv_style=arxiv_style, include_options=include_options,
                    zero_shot_CoT=zero_shot_CoT, few_shot=few_shot,
                    include_abs=include_abs, use_attention=use_attention,
                    options=options, explain=explain, initial_judgement=initial_judgement
                )
                # Write result to a JSON file
                output_data = {
                    "result": result,
                    "reason": reason,
                    "result2": result2
                }
                result_containers[i] = [result, reason, result2]
                with open(f'/data2/wanghongyi/wisconsin/result_{i}.json', 'w') as f:
                    json.dump(output_data, f)
                success = True
            except Exception as e:
                retry_count += 1
                print(f"Exception at index {i}: {e}")
                if retry_count >= retry_limit:
                    # Write exception to a JSON file
                    output_data = {
                        "exception": str(e)
                    }
                    exception_containers[i] = e
                    result_containers[i] = [get_matched_option(initial_judgement[1][node_index_list[i]], options)==text['label'][node_index_list[i]], initial_judgement[0][node_index_list[i]], initial_judgement[1][node_index_list[i]]]
                    with open(f'/data2/wanghongyi/wisconsin/exception_{i}.json', 'w') as f:
                        json.dump(output_data, f)
            except TimeoutError:
                retry_count += 1
                print(f"Timeout at index {i}")
                if retry_count >= retry_limit:
                    output_data = {
                        "exception": "TimeoutError"
                    }
                    exception_containers[i] = e
                    with open(f'/data2/wanghongyi/wisconsin/exception_{i}.json', 'w') as f:
                        json.dump(output_data, f)

    # Control concurrency using ThreadPoolExecutor with max_workers=10
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = []
        for i in range(len(node_index_list)):
        #for i in [5875,6268,7163,7382,7755,10608,11012,11186,12036,13332]:
            futures.append(executor.submit(process_node, i))

        # Optionally, wait for all futures to complete
        for future in as_completed(futures):
            pass  # Results are written to files; no need to collect them here

    # Process results after all threads have completed
    for i in range(len(node_index_list)):
    #for i in [5875,6268,7163,7382,7755,10608,11012,11186,12036,13332]:
        if i in exception_containers and exception_containers[i] is MessageTooLongError:
            too_long_indexes.append(i)
            print(f"Message too long at index {i}")
            #result_containers[i] = [products_keys_list[0]==text['label'][node_index_list[i]], f"Message too long at index {i}", products_keys_list[0]]
        
        if i in result_containers and result_containers[i][0] is not None:
            result, reason, result2 = result_containers[i]
            count += result
            if result==0:
                wrong_indexes.append(i)
           
            if explain:
                explainations.append(result2)
            else:
                results.append(result2)
                wrong_reason.append(reason)

    print("Accuracy:", count/len(node_index_list))
    print("Wrong indexes:", wrong_indexes)
    print("Wrong indexes length:", len(wrong_indexes))
    pickle.dump(results, open(f"wrong_reason.pkl", "wb"))
    assert len(wrong_indexes) == len(node_index_list) - count

    return count/len(node_index_list), wrong_indexes

