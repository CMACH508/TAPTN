"""
Actor dataset loader for TAPE-main.

兼容 load.py 的两值返回格式：get_raw_text_actor(use_text, seed) -> (data, text_list)。

节点集合策略（与 WebKB 保持一致）
─────────────────────────────────
Actor 原始数据集含固定 train/val/test mask。TAPE 仅使用 **原始 test 集**
（889 个节点）作为 LLM 标注节点池，再按 seed 随机 60/20/20 切分为：
  · data.train_id / train_mask  ← TAPE 训练集（60 %）
  · data.val_id   / val_mask    ← TAPE 验证集（20 %）
  · data.test_id  / test_mask   ← TAPE 测试集（20 %）
原始 test_id 保存在 data.actor_original_test_id，供 pkl 索引使用。

Initial Classification and Reasoning: 采用与 LLM-Structured-Data-main 相同的
多通道叙述模版（build_multichannel_judgement），通道文件与 run_taptn.py 调试
模式保持一致。
Refined Classification and Reasoning: 使用 iter2 预测 pkl。
"""

import os
import sys
import copy
import pickle
import random
import numpy as np
import torch

# Vendor helpers live under TAPTN_ASSETS/vendor/data1 (see the asset-bundle README).
_ASSETS = os.environ.get('TAPTN_ASSETS', '')
if not _ASSETS:
    raise RuntimeError(
        'TAPTN_ASSETS is not set. Unpack the asset bundle and export '
        'TAPTN_ASSETS=/path/to/TAPTN_finetune_repro_assets (see README).')
_PROJ_ROOT = os.path.join(_ASSETS, 'vendor', 'data1')
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from utils.load_actor import get_raw_text_actor as _get_raw_text_actor
from utils.utils import sample_test_nodes, build_multichannel_judgement

# ── Actor 类别（用于 majority-vote 提取） ───────────────────────────────────
ACTOR_OPTIONS = {
    'American film actors (only)',
    'American film actors and American television actors',
    'American television actors and American stage actors',
    'English actors',
    'Canadian actors',
}

# ── 默认通道 pkl 文件（与 run_taptn.py 调试配置保持一致） ─────────────────────
_PKL_ROOT = os.path.join(_ASSETS, 'pkls')
DEFAULT_CHANNEL_PKLS = [
    os.path.join(_PKL_ROOT, 'actor_hop1_iter1_neighbors_instr_llama_3.3_70b_i_5.pkl'),
    os.path.join(_PKL_ROOT, 'actor_hop1_iter1_neighbors_instr_refine_llama_3.3_70b_i.pkl'),
    os.path.join(_PKL_ROOT, 'actor_hop1_iter1_neighbors_instr_llama_3.3_70b_i_tn1_2.pkl'),
    os.path.join(_PKL_ROOT, 'actor_hop1_iter1_neighbors_llama_3.3_70b_i.pkl'),
]
DEFAULT_ITER2_PKL = os.path.join(
    _PKL_ROOT, 'actor_hop1_iter2_neighbors_llama_3.3_70b_i.pkl'
)
DEFAULT_CHANNELS_NODE_SET = 'test_and_1hop'
DEFAULT_SHOW_CONSENSUS = False


# ────────────────────────────────────────────────────────────────────────────
# 内部辅助：对原始 test 集按 seed 做 60 / 20 / 20 重新切分
# ────────────────────────────────────────────────────────────────────────────
def _resplit_on_test(data, seed):
    """
    取原始 Actor test_id 作为节点池，以 seed 打乱后按 60/20/20 切分，
    重写 data 的 train/val/test 属性，同时将原始 test_id 保存在
    data.actor_original_test_id 以供 pkl 索引使用。

    返回修改后的 data（in-place）。
    """
    orig_test_id = (data.test_id.numpy()
                    if hasattr(data.test_id, 'numpy') else np.array(data.test_id))

    # 用 numpy RandomState 以保证与 random.seed / np.random.seed 一致的行为
    rng = np.random.default_rng(seed)
    shuffled = orig_test_id.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * 0.6)
    n_val   = int(n * 0.8)   # 60 %~80 % 为 val

    new_train_id = np.sort(shuffled[:n_train])
    new_val_id   = np.sort(shuffled[n_train:n_val])
    new_test_id  = np.sort(shuffled[n_val:])

    n_nodes = data.num_nodes
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask   = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask  = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[new_train_id] = True
    val_mask[new_val_id]     = True
    test_mask[new_test_id]   = True

    # 保存原始 test_id（用于后续 pkl 对齐）
    data.actor_original_test_id = orig_test_id

    data.train_id   = new_train_id
    data.val_id     = new_val_id
    data.test_id    = new_test_id
    data.train_mask = train_mask
    data.val_mask   = val_mask
    data.test_mask  = test_mask

    print(f'[load_actor] TAPE 节点集（重新切分自原始 test={n}）: '
          f'train={len(new_train_id)}, val={len(new_val_id)}, '
          f'test={len(new_test_id)}')
    return data


# ────────────────────────────────────────────────────────────────────────────
# 内部辅助：从 iter2 pkl 构建全图索引数组
# ────────────────────────────────────────────────────────────────────────────
def _build_iter2_by_node(iter2_pkl_path, data_for_lookup, text):
    """
    将 iter2 pkl 的 wrong_reason 映射到全图索引数组（长度=num_nodes）。
    iter2 使用原始 test 集，data_for_lookup.test_id 必须为原始 test_id。
    非 test 节点填 None。
    """
    with open(iter2_pkl_path, 'rb') as f:
        pk = pickle.load(f)

    results      = pk['results']
    wrong_reason = pk.get('wrong_reason', [None] * len(results))

    nil = sample_test_nodes(data_for_lookup, text, None, 'actor', node_set='test')
    assert len(nil) == len(results), (
        f'[load_actor] iter2 node_index_list len {len(nil)} != results {len(results)}'
    )

    reason_by_node = [None] * data_for_lookup.num_nodes
    result_by_node = [None] * data_for_lookup.num_nodes
    for pos, nidx in enumerate(nil):
        reason_by_node[nidx] = wrong_reason[pos]
        result_by_node[nidx] = results[pos]

    return reason_by_node, result_by_node


# ────────────────────────────────────────────────────────────────────────────
# 公开接口
# ────────────────────────────────────────────────────────────────────────────
def get_raw_text_actor(use_text=False, seed=0):
    """
    返回 (data, text_list)，兼容 TAPE-main load.py 的二值接口。

    · data 的 train/val/test 已按 seed 在原始 test 集上重新切分（60/20/20）。
    · data.actor_original_test_id 保存原始 test_id，供 build_actor_gpt_text 使用。
    · text_list[i] 为节点 i 的纯文本描述（str）；use_text=False 时为 None。
    """
    data, _data2, text = _get_raw_text_actor(use_text=use_text, seed=seed)

    # 在原始 test 集上重新切分
    data = _resplit_on_test(data, seed)

    if not use_text:
        return data, None

    text_list = [str(nt) for nt in text['node_text']]
    return data, text_list


def build_actor_gpt_text(
    data,
    text_list,
    text_dict=None,
    channel_pkls=None,
    iter2_pkl=None,
    channels_node_set=None,
    show_consensus=None,
    seed=0,
):
    """
    为 use_gpt=True 分支构建带 LLM 推理的文本列表。

    参数
    ----
    data        : get_raw_text_actor 返回的 PyG Data（已重新切分）
    text_list   : 各节点的纯文本列表（长度=num_nodes）
    text_dict   : 原始 text dict（含 'label', 'node_text' 等）。
                  若为 None，将自动调用原始 Actor 加载器获取。
    channel_pkls: iter1 通道 pkl 路径列表（默认 DEFAULT_CHANNEL_PKLS）
    iter2_pkl   : iter2 pkl 路径（默认 DEFAULT_ITER2_PKL）
    channels_node_set: channel pkl 生成时使用的节点集（默认 DEFAULT_CHANNELS_NODE_SET）
    show_consensus   : 是否显示 Consensus 行（默认 DEFAULT_SHOW_CONSENSUS）
    seed        : 仅在自动获取 text_dict 时使用（默认 0）

    返回
    ----
    text2 : list[str]，每个元素为一个节点的完整 TAPE 输入文本
    """
    # 若未传入 text_dict，从原始 Actor 加载器自动获取
    if text_dict is None:
        print('[load_actor] 自动获取 text_dict ...')
        _, _, text_dict = _get_raw_text_actor(use_text=True, seed=seed)
    if channel_pkls    is None: channel_pkls    = DEFAULT_CHANNEL_PKLS
    if iter2_pkl       is None: iter2_pkl       = DEFAULT_ITER2_PKL
    if channels_node_set is None: channels_node_set = DEFAULT_CHANNELS_NODE_SET
    if show_consensus  is None: show_consensus  = DEFAULT_SHOW_CONSENSUS

    n = data.num_nodes

    # ── 构建用于 pkl 索引的"原始 mask"视图 ─────────────────────────────────
    # data 已重新切分；需用原始 test_id 重建 test_mask，才能正确对齐 channel / iter2 pkl
    if hasattr(data, 'actor_original_test_id'):
        orig_test_id = data.actor_original_test_id
    else:
        raise RuntimeError(
            '[load_actor] data.actor_original_test_id 不存在，'
            '请确认通过 get_raw_text_actor() 加载数据。'
        )

    # 浅拷贝 data，还原 test_id / test_mask 供 sample_test_nodes 使用
    data_orig = copy.copy(data)
    orig_test_mask = torch.zeros(n, dtype=torch.bool)
    orig_test_mask[orig_test_id] = True
    data_orig.test_id   = orig_test_id
    data_orig.test_mask = orig_test_mask

    # ── 多通道初始推理 ────────────────────────────────────────────────────
    existing = [p for p in channel_pkls if os.path.exists(p)]
    missing  = [p for p in channel_pkls if not os.path.exists(p)]
    if missing:
        print(f'[load_actor] 警告: {len(missing)} 个 channel pkl 不存在，已跳过:')
        for p in missing: print(f'  {p}')

    combined_reason = [None] * n
    if existing:
        print(f'[load_actor] 构建多通道初始推理（{len(existing)} 个 channel）...')
        combined_reason, _ = build_multichannel_judgement(
            existing, data_orig, text_dict,
            dataset='actor',
            node_set=channels_node_set,
            options=ACTOR_OPTIONS,
            show_consensus=show_consensus,
        )
    else:
        print('[load_actor] 警告: 无可用 channel pkl，Initial Classification 将为空。')

    # ── iter2 精炼推理 ────────────────────────────────────────────────────
    iter2_reason_by_node = [None] * n
    if iter2_pkl and os.path.exists(iter2_pkl):
        print(f'[load_actor] 加载 iter2 pkl: {iter2_pkl}')
        iter2_reason_by_node, _ = _build_iter2_by_node(
            iter2_pkl, data_orig, text_dict
        )
    else:
        print('[load_actor] 警告: iter2 pkl 不存在或未指定，Refined Classification 将为空。')

    # ── 拼装最终文本 ─────────────────────────────────────────────────────
    text2 = []
    for i in range(n):
        content = f'Title and Abstract:\n{text_list[i]}'
        if combined_reason[i] is not None:
            content += (f'\n\nInitial Classification and Reasoning:'
                        f'\n{combined_reason[i]}')
        if iter2_reason_by_node[i] is not None:
            content += (f'\n\nRefined Classification and Reasoning:'
                        f'\n{iter2_reason_by_node[i]}')
        text2.append(content)

    return text2
