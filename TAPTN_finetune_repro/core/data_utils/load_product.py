"""
Product (ogbn-products) dataset loader for TAPE-main.

兼容 load.py 的两值返回格式：get_raw_text_product(use_text, seed) -> (data, text_list)。

两阶段节点集合策略
─────────────────────────────────
阶段 1 ── 固定池选取（seed=42, size=400）
  从原始 test 集用固定种子洗牌，取末尾 400 个节点作为「节点池」
  （对应用户代码中的 _test_idx = _shuffled[800:1200]）。
  池的组成不随训练 seed 变化，保存在 data.product_pool_ids。
  同时将 data.product_pool_mask 设为覆盖这 400 个节点的 Boolean mask，
  供 pkl 对齐（build_product_gpt_text）使用。

阶段 2 ── 运行时重切分（按训练 seed，60/20/20）
  对池中 400 个节点用运行时 seed 洗牌后按 60/20/20 切分：
    · train_mask ← 240 节点
    · val_mask   ← 80  节点
    · test_mask  ← 80  节点
  每个不同的训练 seed 产生不同但稳定可复现的切分。

pkl 对齐
─────────────────────────────────
iter1 / iter2 pkl 均以固定池（400 节点）为索引基准，
build_product_gpt_text 内部使用 data.product_pool_mask 还原池视图，
与运行时切分的 test_mask 无关。

Initial Classification and Reasoning:  单通道 iter1 pkl
  product_hop1_noanon_guide_llama-3.3-70b-instruct_2.pkl
Refined Classification and Reasoning:  iter2 pkl
  product_hop1_noanon_guide_llama-3.3-70b-instruct_iter2_3.pkl
"""

import os
import sys
import copy
import pickle
import builtins
import numpy as np
import torch

# ── 路径策略 ──────────────────────────────────────────────────────────────────
# home 版：包含 get_raw_text_products / products_mapping（无 load_products.py 在 data1）
# data1 版：包含 build_multichannel_judgement / sample_test_nodes（含 node_set 参数）
#
# 插入顺序：先插 home，再插 data1（data1 最终排在 sys.path 最前面），
# 使 `from utils.utils import ...` 优先找到 data1 版；
# `from utils.load_products import ...` 在 data1 中不存在，自动回落到 home 版。
_ASSETS = os.environ.get('TAPTN_ASSETS', '')
_HOME_ROOT = os.path.join(_ASSETS, 'vendor', 'home') if _ASSETS else ''
_PROJ_ROOT = os.path.join(_ASSETS, 'vendor', 'data1') if _ASSETS else ''
_DATA_ROOT = _ASSETS
_PKL_BASE = os.path.join(_ASSETS, 'pkls') if _ASSETS else 'pkls'
_OGB_DATASET_ROOT = os.path.join(_ASSETS, 'dataset') if _ASSETS else 'dataset'
if _HOME_ROOT and _HOME_ROOT not in sys.path:
    sys.path.insert(0, _HOME_ROOT)
if _PROJ_ROOT and _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

# Vendor helpers are imported lazily. Dry-run with product_cache must not import
# vendor utils.py (it pulls OpenAI clients and PyTorch 2.6-only APIs).
_get_raw_text_products = None
products_mapping = None
products_keys_list = None
_CATEGORY_TO_IDX = None
PRODUCT_OPTIONS = None


def _get_matched_option(prediction, valid_options):
    prediction = prediction.lower()
    matched_option = ""
    earliest_position = len(prediction)
    for option in valid_options:
        position = prediction.find(option.lower())
        if position != -1 and position < earliest_position:
            matched_option = option
            earliest_position = position
    if matched_option == "Computational Learning Theory":
        matched_option = "Theory"
    return matched_option


def _ensure_product_vendor():
    """Load ogbn-products helpers only when the on-disk cache is missing."""
    global _get_raw_text_products, products_mapping, products_keys_list
    global _CATEGORY_TO_IDX, PRODUCT_OPTIONS
    if _get_raw_text_products is not None:
        return
    import utils.load_products as _lp_module
    from utils.load_products import (
        get_raw_text_products as _fn,
        products_mapping as _pm,
        products_keys_list as _pkl,
    )
    from ogb.nodeproppred import PygNodePropPredDataset as _OrigPygDataset

    def _PygDatasetWithRoot(name, root=_OGB_DATASET_ROOT, **kwargs):
        return _OrigPygDataset(name=name, root=root, **kwargs)

    _lp_module.PygNodePropPredDataset = _PygDatasetWithRoot
    _get_raw_text_products = _fn
    products_mapping = _pm
    products_keys_list = _pkl
    _CATEGORY_TO_IDX = {name: idx for idx, name in enumerate(_pkl)}
    PRODUCT_OPTIONS = set(_pkl)


def _category_to_idx():
    global _CATEGORY_TO_IDX
    if _CATEGORY_TO_IDX is None:
        _ensure_product_vendor()
    return _CATEGORY_TO_IDX

# ── 默认 pkl 路径（_PKL_BASE 已在上方按 TAPTN_ASSETS 设定）─────────────────────
# sem=True（结构引导，guided）
DEFAULT_CHANNEL_PKLS = [
    os.path.join(_PKL_BASE, 'product_hop1_noanon_guide_llama-3.3-70b-instruct_2.pkl'),
]
DEFAULT_ITER2_PKL = os.path.join(
    _PKL_BASE, 'product_hop1_noanon_guide_llama-3.3-70b-instruct_iter2_3.pkl'
)

# sem=False（无结构引导，noguide；单通道，无 iter2）
DEFAULT_NOSEM_CHANNEL_PKLS = [
    os.path.join(_PKL_BASE, 'product_hop2_noanon_noguide_llama-3.3-70b-instruct.pkl'),
]
DEFAULT_NOSEM_ITER2_PKL = ''   # 空字符串表示不使用 iter2

# pkl 生成时以固定池（400 节点）为 test_mask；对应 build_multichannel_judgement
# 中的 node_set——使用 data_pool 视图（test_mask = 400 节点）时填 'test'
DEFAULT_CHANNELS_NODE_SET = 'test'
DEFAULT_SHOW_CONSENSUS = False

# ── 文本格式优化参数 ──────────────────────────────────────────────────────────
# product 数据集每段 LLM 推理文本的平均 token 数：
#   iter1 reason: ~664 tokens，iter2 reason: ~630 tokens，plain: ~169 tokens
#   合计 ~1526 tokens → 平均 4.5 个滑窗 chunk（chunk=512, stride=256）
#
# 问题：iter2 的高精度分类结论（82%+）位于文本末尾（56% token位置），
#       被均值聚合后权重仅占 ~22%，远低于无效内容 chunk。
# 优化策略：
#   1. 结论前置（result-first）：将分类结论放在推理文本前面，确保在第一个 chunk 可见
#   2. 截断推理：限制每段 reason 的字符数，避免长推理导致关键结论被推至末尾
#
# REASON_TRUNC_CHARS：每段 reason 保留的最大字符数（≈ chars/4 tokens）。
#   - None 或 0：不截断（使用原始完整推理文本，保留原始语义但 chunk 数较多）
#   - 600：约150 tokens，截断后 avg total tokens ~750，多数文本在 2 个 chunk 内
#   - 1200：约300 tokens，截断后 avg total tokens ~1050，avg ~3 chunk
# 默认 0（不截断），保持向后兼容；可通过 config.product.reason_trunc_chars 覆盖。
REASON_TRUNC_CHARS = 1200  # ≈300 tokens；实测79%样本iter2标签进入chunk0（可提升至91%设600）
# 设为 0 可禁用截断（向后兼容旧行为，但 iter2 信号仅 5% 在 chunk0）

# ── 阶段 1 固定参数 ──────────────────────────────────────────────────────────
PRODUCT_POOL_SEED = 42    # 固定池选取种子（不随训练 seed 变化）
PRODUCT_POOL_SIZE = 400   # 池大小（pkl 覆盖的节点数）

# ── 磁盘缓存（避免每次 seed 重复 5-10 分钟的 2.8GB 加载 + LCC 图计算） ────────
# 缓存保存 _get_raw_text_products 执行完毕后、TAPE 切分前的 data 对象；
# text_dict 单独缓存（仅 use_gpt=True 时才需要）。
_PRODUCT_CACHE_DIR  = os.path.join(_DATA_ROOT, 'dataset', 'product_cache')
_PRODUCT_DATA_CACHE = os.path.join(_PRODUCT_CACHE_DIR, 'product_lcc_data.pt')
_PRODUCT_TEXT_CACHE = os.path.join(_PRODUCT_CACHE_DIR, 'product_text_dict.pkl')


def _load_product_base(use_text: bool) -> tuple:
    """
    返回 (data, text_dict)，data 为 LCC 后、池选取前的原始 PyG Data 对象。

    策略：
      - data 一旦缓存到 _PRODUCT_DATA_CACHE，后续直接从文件加载（秒级）。
      - text_dict 同理缓存到 _PRODUCT_TEXT_CACHE，仅 use_text=True 时加载。
      - 首次运行需要 ~5-10 分钟（2.8GB 文件加载 + 图计算 + LCC）。
    """
    # ── 加载 data ──────────────────────────────────────────────────────────
    if os.path.exists(_PRODUCT_DATA_CACHE):
        print(f'[load_product] 从磁盘缓存加载 data: {_PRODUCT_DATA_CACHE}')
        data = torch.load(_PRODUCT_DATA_CACHE, weights_only=False)
    else:
        print('[load_product] 首次运行：加载 2.8GB ogbn-products + LCC (约5-10分钟)...')
        _orig_cwd   = os.getcwd()
        _orig_input = builtins.input
        builtins.input = lambda prompt='': (
            print(f'[load_product] OGB prompt auto-answered N: {prompt}') or 'N'
        )
        try:
            os.chdir(_HOME_ROOT)
            _ensure_product_vendor()
            data, _ = _get_raw_text_products(use_text=False)
        finally:
            os.chdir(_orig_cwd)
            builtins.input = _orig_input
        os.makedirs(_PRODUCT_CACHE_DIR, exist_ok=True)
        torch.save(data, _PRODUCT_DATA_CACHE)
        print(f'[load_product] data 缓存已写入: {_PRODUCT_DATA_CACHE}')

    # ── 加载 text_dict（可选） ────────────────────────────────────────────
    text_dict = None
    if use_text:
        if os.path.exists(_PRODUCT_TEXT_CACHE):
            print(f'[load_product] 从磁盘缓存加载 text_dict: {_PRODUCT_TEXT_CACHE}')
            with open(_PRODUCT_TEXT_CACHE, 'rb') as f:
                text_dict = pickle.load(f)
        else:
            print('[load_product] 首次运行：加载 text_dict (Amazon-3M JSON, 需数分钟)...')
            _orig_cwd   = os.getcwd()
            _orig_input = builtins.input
            builtins.input = lambda prompt='': (
                print(f'[load_product] OGB prompt auto-answered N: {prompt}') or 'N'
            )
            try:
                os.chdir(_HOME_ROOT)
                _ensure_product_vendor()
                _, text_dict = _get_raw_text_products(use_text=True)
            finally:
                os.chdir(_orig_cwd)
                builtins.input = _orig_input
            os.makedirs(_PRODUCT_CACHE_DIR, exist_ok=True)
            with open(_PRODUCT_TEXT_CACHE, 'wb') as f:
                pickle.dump(text_dict, f)
            print(f'[load_product] text_dict 缓存已写入: {_PRODUCT_TEXT_CACHE}')

    return data, text_dict


# ────────────────────────────────────────────────────────────────────────────
# 阶段 1：固定池选取
# ────────────────────────────────────────────────────────────────────────────
def _extract_pool(data):
    """
    用固定种子（PRODUCT_POOL_SEED=42）从原始 test 节点中提取 400 节点池。

    对应用户代码逻辑：
        _shuffled = rng.permutation(_orig_test_idx)   # seed=42
        pool      = _shuffled[800:1200]               # 最后 400 个

    写入 data 的属性：
        data.product_pool_ids  : np.ndarray, 形状 (400,)，池节点的全局索引
        data.product_pool_mask : torch.BoolTensor, 形状 (num_nodes,)，池节点 mask
          ↑ 上述两者供 build_product_gpt_text 做 pkl 对齐，不受运行时 seed 影响

    返回修改后的 data（in-place）。
    """
    orig_test_idx = np.where(data.test_mask.numpy())[0]

    rng = np.random.default_rng(PRODUCT_POOL_SEED)
    shuffled = rng.permutation(orig_test_idx)

    assert len(shuffled) >= 3 * PRODUCT_POOL_SIZE, (
        f'[load_product] 原始 test 节点数 ({len(shuffled)}) 不足以抽取 '
        f'3×{PRODUCT_POOL_SIZE} 个互不重叠节点'
    )

    # 与用户代码一致：取末尾 400 个（索引 800-1200）作为节点池
    pool_ids = shuffled[2 * PRODUCT_POOL_SIZE : 3 * PRODUCT_POOL_SIZE]

    pool_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    pool_mask[pool_ids] = True

    data.product_pool_ids  = np.sort(pool_ids)   # 排序保证确定性
    data.product_pool_mask = pool_mask

    print(f'[load_product] 固定池选取 (seed={PRODUCT_POOL_SEED}): '
          f'{len(pool_ids)} 个节点')
    return data


# ────────────────────────────────────────────────────────────────────────────
# 阶段 2：运行时重切分（60 / 20 / 20）
# ────────────────────────────────────────────────────────────────────────────
def _resplit_pool(data, seed):
    """
    对 data.product_pool_ids（400 节点）按运行时 seed 做 60/20/20 切分，
    重写 data 的 train/val/test mask 和 *_id 属性。

    切分结果（默认）：
        train_mask  240 节点（60 %）
        val_mask     80 节点（20 %）
        test_mask    80 节点（20 %）

    data.product_pool_mask 不受影响，始终指向全部 400 个池节点。

    返回修改后的 data（in-place）。
    """
    if not hasattr(data, 'product_pool_ids'):
        raise RuntimeError(
            '[load_product] data.product_pool_ids 不存在，请先调用 _extract_pool()。'
        )

    pool = data.product_pool_ids   # np.ndarray, sorted, 400 个全局节点索引

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(pool)

    n_pool  = len(shuffled)
    n_train = int(n_pool * 0.6)
    n_val   = int(n_pool * 0.8)   # 60 %~80 % 为 val

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

    data.train_id   = new_train_id
    data.val_id     = new_val_id
    data.test_id    = new_test_id
    data.train_mask = train_mask
    data.val_mask   = val_mask
    data.test_mask  = test_mask

    print(f'[load_product] 运行时重切分 (seed={seed}, pool={n_pool}): '
          f'train={len(new_train_id)}, val={len(new_val_id)}, '
          f'test={len(new_test_id)}')
    return data


# ────────────────────────────────────────────────────────────────────────────
# 阶段 1.5：LCC 紧缩
# ────────────────────────────────────────────────────────────────────────────
def _compact_to_lcc(data):
    """
    将 2.4M 节点全图紧缩为 LCC 子图（约 13482 节点）。

    调用时机：_extract_pool 之后、_resplit_pool 之前。

    此时 data.test_mask  = LCC 成员 mask（2449029 长）
         data.test_id    = LCC 成员全局索引（sorted tensor/array）
         data.product_pool_ids  = 400 个全局索引（需重映射）
         data.product_pool_mask = 2449029 长布尔 mask（需重映射）

    紧缩后：
        data.num_nodes         = N_lcc（约 13482）
        data.x / y             = sliced 到 LCC 节点
        data.edge_index        = 重编号到 0..N_lcc-1
        data.train/val/test_mask = 原 OGB mask 切片（仅保留 LCC 节点）
        data.product_pool_ids  = LCC-local 索引（400 个）
        data.product_pool_mask = N_lcc 长布尔 mask（400 True）
        data.lcc_global_ids    = 原始全局索引 tensor（供 text_dict 查阅）

    关于 pkl 对齐：
        sample_test_nodes 返回 np.where(test_mask)[0]，即升序索引。
        LCC-local 升序 ≡ global 升序（local_i = rank of global_i 在排序后 LCC 列表中），
        因此 pkl 中节点的排列顺序不变，对齐逻辑正确。
    """
    # ── 获取 LCC 全局索引 ────────────────────────────────────────────────
    if hasattr(data, 'test_id') and len(getattr(data, 'test_id', [])) > 400:
        # test_id 此时仍为 LCC transform 设置的全局索引（未被 _resplit_pool 覆写）
        lcc_global = data.test_id
    else:
        # 退路：从 LCC membership mask 推导
        lcc_global = torch.where(data.test_mask)[0]

    if isinstance(lcc_global, np.ndarray):
        lcc_global = torch.from_numpy(lcc_global.astype(np.int64))
    lcc_global = lcc_global.long()
    n_lcc = len(lcc_global)

    # ── 建立 global → local 反向映射 ────────────────────────────────────
    remap = torch.full((data.num_nodes,), -1, dtype=torch.long)
    remap[lcc_global] = torch.arange(n_lcc)

    # ── 重编号 edge_index ─────────────────────────────────────────────
    new_edge_index = remap[data.edge_index]
    assert (new_edge_index >= 0).all(), '[compact_to_lcc] edge_index 含非 LCC 节点'

    # ── 构建紧缩 Data 对象 ────────────────────────────────────────────
    from torch_geometric.data import Data as _PyGData
    compact = _PyGData()
    compact.num_nodes   = n_lcc
    compact.edge_index  = new_edge_index
    compact.lcc_global_ids = lcc_global   # 保留，供 text_dict 查询

    for attr in ('x', 'y'):
        val = getattr(data, attr, None)
        if val is not None and len(val) == data.num_nodes:
            setattr(compact, attr, val[lcc_global])

    for attr in ('train_mask', 'val_mask', 'test_mask'):
        val = getattr(data, attr, None)
        if val is not None and len(val) == data.num_nodes:
            setattr(compact, attr, val[lcc_global])

    for attr in ('in_degree', 'out_degree'):
        val = getattr(data, attr, None)
        if val is not None and len(val) == data.num_nodes:
            setattr(compact, attr, val[lcc_global])

    # ── 重映射 pool 信息（global → local）────────────────────────────
    if hasattr(data, 'product_pool_ids'):
        pool_global = torch.from_numpy(
            np.asarray(data.product_pool_ids, dtype=np.int64))
        pool_local  = remap[pool_global].numpy()
        assert (pool_local >= 0).all(), '[compact_to_lcc] pool 节点不在 LCC 中'
        compact.product_pool_ids = np.sort(pool_local)

    if hasattr(data, 'product_pool_mask'):
        old_mask = data.product_pool_mask
        if len(old_mask) == data.num_nodes:
            compact.product_pool_mask = old_mask[lcc_global]
        else:
            compact.product_pool_mask = old_mask   # 已是 local，不需要切片

    print(f'[load_product] LCC 紧缩: {data.num_nodes} → {n_lcc} 节点，'
          f'edges={new_edge_index.shape[1]}')
    return compact


# ────────────────────────────────────────────────────────────────────────────
# 内部辅助：将 text dict 转成 list[str]（长度=num_nodes）
# ────────────────────────────────────────────────────────────────────────────
def _make_text_list(data, text_dict):
    """
    将 get_raw_text_products 返回的 text dict 转换为每节点一个字符串的列表。
    title / content 缺失时用空字符串占位。

    若 data 已紧缩（具有 lcc_global_ids），则用全局索引查 text_dict，
    否则直接用本地节点序号（适用于未紧缩的全图）。
    """
    n = data.num_nodes
    titles   = text_dict.get('title',   []) or []
    contents = text_dict.get('content', []) or []

    # 使用全局索引查 text_dict（紧缩数据）还是本地索引（全图数据）
    lcc_global_ids = getattr(data, 'lcc_global_ids', None)

    text_list = []
    for i in range(n):
        g = int(lcc_global_ids[i]) if lcc_global_ids is not None else i
        title   = (titles[g]   if g < len(titles)   else None) or ''
        content = (contents[g] if g < len(contents) else None) or ''
        text_list.append(f'Title: {title}\nContent: {content}')
    return text_list


# ────────────────────────────────────────────────────────────────────────────
# 内部辅助：从 product pkl（{global_idx: str} 格式）构建 LCC-local 数组
# ────────────────────────────────────────────────────────────────────────────
# product pkl 格式与 actor/arxiv 不同：
#   pkl['result'] = {global_node_idx: prediction_str, ...}
#   pkl['reason'] = {global_node_idx: reasoning_str,  ...}
# 无需 sample_test_nodes 位置对齐，直接用全局索引查字典即可。
# ─────────────────────────────────────────────────────────────────────────────

def _pkl_split_by_lcc(pkl_path, lcc_global_ids):
    """
    读取一个 product pkl，分别返回 reason 列表和 result 列表，
    长度均为 len(lcc_global_ids)。不在 pkl 中的节点填 None。

    pkl 格式：{'result': {global_idx: str}, 'reason': {global_idx: str}, ...}

    返回
    ----
    reasons : list[str|None]
    results : list[str|None]
    """
    with open(pkl_path, 'rb') as f:
        pk = pickle.load(f)

    reason_dict = pk.get('reason', {})
    result_dict = pk.get('result', {})

    reasons, results = [], []
    for g in lcc_global_ids:
        g = int(g)
        reasons.append(reason_dict.get(g, None))
        results.append(result_dict.get(g, None))
    return reasons, results


def _pkl_reason_by_lcc(pkl_path, lcc_global_ids):
    """
    [向后兼容] 读取一个 product pkl，返回长度=len(lcc_global_ids) 的合并字符串列表。
    新代码请使用 _pkl_split_by_lcc 获取独立的 reason / result。
    """
    reasons, results = _pkl_split_by_lcc(pkl_path, lcc_global_ids)
    out = []
    for r, s in zip(reasons, results):
        if r or s:
            parts = []
            if r: parts.append(str(r))
            if s: parts.append(f'Classification: {s}')
            out.append('\n'.join(parts))
        else:
            out.append(None)
    return out


def _iter2_reason_by_lcc(iter2_pkl_path, lcc_global_ids):
    """
    [向后兼容] 读取 iter2 pkl，返回合并字符串列表。
    新代码请使用 _pkl_split_by_lcc。
    """
    return _pkl_reason_by_lcc(iter2_pkl_path, lcc_global_ids)


def _build_iter2_by_node(iter2_pkl_path, data_pool_view, text_dict):
    """已废弃（product pkl 格式不兼容）；保留签名以防其他代码引用。"""
    raise RuntimeError(
        '[load_product] _build_iter2_by_node 已被 _iter2_reason_by_lcc 替代，'
        '请勿直接调用。'
    )


# ────────────────────────────────────────────────────────────────────────────
# 公开接口 1：基础数据加载
# ────────────────────────────────────────────────────────────────────────────
def get_raw_text_product(use_text=False, seed=0):
    """
    返回 (data, text_list)，兼容 TAPE-main load.py 的二值接口。

    执行流程：
      1. 调用上游 get_raw_text_products 获取 PyG Data
      2. 阶段1：固定池选取（seed=42），写入 product_pool_ids / product_pool_mask
      3. 阶段2：运行时重切分（按 seed 参数），写入 train/val/test mask
         · pool 400 节点 → 240 train / 80 val / 80 test

    · data.product_pool_ids  : 固定的 400 节点全局索引（排序），供 pkl 对齐
    · data.product_pool_mask : 覆盖 400 节点的 Boolean mask，供 pkl 对齐
    · text_list[i]           : 节点 i 的 "Title: ...\nContent: ..." 字符串；
                               use_text=False 时返回 None
    """
    # 从磁盘缓存（或首次构建）获取 data / text_dict，
    # 避免每次 seed 都重复 2.8GB 加载 + LCC 图计算（约 5-10 分钟）。
    import copy as _copy
    data, text_dict = _load_product_base(use_text=use_text)
    # 浅拷贝：_extract_pool 只添加属性，不做就地张量修改，不污染缓存对象。
    data = _copy.copy(data)

    # 阶段 1：固定池选取（全局索引，与 seed 无关，与 home 版对齐）
    data = _extract_pool(data)

    # 阶段 1.5：LCC 紧缩（2449029 → ~13482 节点）
    #   ‣ 必须在 _extract_pool 之后执行，确保 product_pool_ids 先以全局索引确定
    #   ‣ 在 _resplit_pool 之前执行，使后续 mask 操作在紧缩空间内进行
    #   ‣ lcc_global_ids 保存在 data 中，供 _make_text_list 查阅 text_dict
    data = _compact_to_lcc(data)

    # 阶段 2：运行时重切分（随 seed 变化，现在在紧缩空间内操作）
    data = _resplit_pool(data, seed)

    if not use_text:
        return data, None

    # text_list 长度 = LCC 节点数（~13482），用 lcc_global_ids 映射 text_dict
    text_list = _make_text_list(data, text_dict)
    return data, text_list


# ────────────────────────────────────────────────────────────────────────────
# 公开接口 2：构建带 LLM 推理的文本列表（use_gpt=True 分支）
# ────────────────────────────────────────────────────────────────────────────
# split_seg 模式使用的段落分隔符（必须与 lm_trainer.py 中保持一致）。
# 选用 ASCII \x00 + 可读标识，确保不会出现在 LLM 输出文本中。
SEG_SEP = "\x00===SEG===\x00"

_VALID_TEXT_FMTS = ('result_first', 'reason_first', 'split_seg')


def _extract_llm_label_idx(result_str: str) -> int:
    """
    从 LLM 原始结论字符串中提取类别索引（对应 data.y 整数标签）。

    使用 get_matched_option 做最早匹配，返回对应的 OGB 标签索引。
    无法匹配时返回 -1（后续转为全零 one-hot，不提供任何类别先验）。

    示例：
      "$\\boxed{Books:1.0}$"        → 4   (Books)
      "Clothing, Shoes & Jewelry (0.9)"  → 11
      None / ""                          → -1
    """
    if not result_str:
        return -1
    matched = _get_matched_option(str(result_str), set(_category_to_idx()))
    if not matched:
        return -1
    return _category_to_idx().get(matched, -1)


def build_product_gpt_text(
    data,
    text_list,
    text_dict=None,
    channel_pkls=None,
    iter2_pkl=None,
    channels_node_set=None,
    show_consensus=None,
    sem=True,
    seed=0,
    reason_trunc_chars=None,
    title_trunc_chars=0,   # title+abstract 截断字符数（0=不截断）；用于防御极端异常值
    text_fmt=None,
    include_iter1=True,    # 是否展示 iter1/初始推理段（sem=False 时忽略，始终展示单一来源）
    merge_cls_reason=False, # split_seg 专用：结论+推理合并为同一段（方案B）
):
    """
    为 use_gpt=True 分支构建带 LLM 推理的文本列表。

    ── text_fmt：仅控制文本排布结构 ──────────────────────────────────────────
    'result_first'  (默认/新格式）
        结论前置：分类标签紧跟 section 标题，推理文本在后，支持字符截断。

    'reason_first'  (旧格式/向后兼容）
        推理在前，分类标签在后，忠实还原旧版（actor/product 原始）行为。

    'split_seg'  (强制分离格式）
        result / reason / 不同 iter 各占独立 chunk（SEG_SEP 分隔）。
        lm_trainer 检测到后对每段独立 tokenize 到 512 token，stride=512。

    ── sem + include_iter1：控制内容范围（独立于 text_fmt）──────────────────
    sem=True  + include_iter1=True  (默认）
        guided pkls (iter1 + iter2) 均展示：
          Initial Classification...  ← iter1（channel pkls）
          Refined Classification...  ← iter2 pkl

    sem=True  + include_iter1=False
        跳过 iter1 段，仅展示 iter2：
          Refined Classification...  ← iter2 pkl

    sem=False（noguide，单一来源）
        include_iter1 被忽略；noguide pkl 始终展示，标签固定为：
          Expert Classification...   ← nosem_channel_pkls

    ── 输出格式示例（result_first, sem=True, include_iter1=True）────────────
        Title and Abstract: ...
        Initial Classification: {iter1_result}
        Reasoning: {iter1_reason[:trunc]}
        Refined Classification: {iter2_result}
        Reasoning: {iter2_reason[:trunc]}

    ── 输出格式示例（result_first, sem=False）───────────────────────────────
        Title and Abstract: ...
        Expert Classification: {nosem_result}
        Reasoning: {nosem_reason[:trunc]}

    参数
    ----
    data              : get_raw_text_product 返回的 PyG Data
    text_list         : 各节点纯文本列表（长度=num_nodes）
    text_dict         : 原始 text dict；None 时自动走磁盘缓存
    channel_pkls      : iter1 通道 pkl 列表；None → 按 sem 选默认值
    iter2_pkl         : iter2 pkl 路径；None → 按 sem 选默认值
    channels_node_set : 保留参数（暂不使用）
    show_consensus    : 保留参数（暂不使用）
    sem               : True=guided / False=noguide（决定数据来源及标签）
    seed              : 仅在自动获取 text_dict 时使用
    reason_trunc_chars: reason 最大字符数；None→模块级 REASON_TRUNC_CHARS；0=不截断
    text_fmt          : 文本排布格式（仅结构）；None→'result_first'
    include_iter1     : 是否展示 iter1 段；sem=False 时此参数无效
    merge_cls_reason  : split_seg 专用。True=结论+推理合并为一段（方案B，避免退化 chunk）；
                        False=结论/推理各占独立 chunk（原始行为）。

    返回
    ----
    text2 : list[str]，每个元素为一个节点的完整 TAPE 输入文本。
    副作用: 在 data 上设置 data.llm_label_idx (torch.LongTensor, shape=[num_nodes])，
           记录每个 LCC 节点从 LLM 结论中提取的类别索引（-1=未知）。
           可被 lm_trainer 读取并作为 one-hot 辅助输入喂给分类头。
    """
    # ── text_fmt 默认与校验 ────────────────────────────────────────────────
    if text_fmt is None:
        text_fmt = 'result_first'
    if text_fmt not in _VALID_TEXT_FMTS:
        raise ValueError(
            f'[load_product] 不支持的 text_fmt={text_fmt!r}，'
            f'可选值: {_VALID_TEXT_FMTS}'
        )

    # ── 根据 sem 选择默认 pkl ──────────────────────────────────────────────
    if channel_pkls is None:
        channel_pkls = DEFAULT_CHANNEL_PKLS if sem else DEFAULT_NOSEM_CHANNEL_PKLS
    if iter2_pkl is None:
        iter2_pkl = DEFAULT_ITER2_PKL if sem else DEFAULT_NOSEM_ITER2_PKL

    # ── 获取 text_dict（若未传入） ────────────────────────────────────────
    if text_dict is None:
        print('[load_product] 自动获取 text_dict (走磁盘缓存)...')
        _, text_dict = _load_product_base(use_text=True)

    if channels_node_set is None: channels_node_set = DEFAULT_CHANNELS_NODE_SET
    if show_consensus    is None: show_consensus    = DEFAULT_SHOW_CONSENSUS
    if reason_trunc_chars is None:
        reason_trunc_chars = REASON_TRUNC_CHARS

    # sem=False 时 include_iter1 无实质意义（单一来源始终展示）
    _show_iter1 = include_iter1 if sem else True

    mode_tag  = 'sem' if sem else 'no_sem'
    incl_tag  = f'include_iter1={_show_iter1}' if sem else 'expert-only'
    trunc_tag = f'trunc={reason_trunc_chars}chars' if reason_trunc_chars else 'no-trunc'
    print(f'[load_product] build_product_gpt_text  '
          f'sem={mode_tag}  fmt={text_fmt}  {incl_tag}  {trunc_tag}')

    # ── 校验 ─────────────────────────────────────────────────────────────
    if not hasattr(data, 'lcc_global_ids'):
        raise RuntimeError(
            '[load_product] data.lcc_global_ids 不存在，'
            '请确认通过 get_raw_text_product() 加载（含 _compact_to_lcc 步骤）。'
        )

    n              = data.num_nodes
    lcc_global_ids = data.lcc_global_ids   # LCC-local i → 全局 node idx

    # ── 标签确定（独立于 text_fmt）────────────────────────────────────────
    # sem=True : iter1 → "Initial ..."  / iter2 → "Refined ..."
    # sem=False: 单一 noguide 来源 → "Expert ..."（不受 include_iter1 影响）
    if sem:
        _iter1_cls_label = 'Initial Classification'
        _iter1_rsn_label = 'Initial Reasoning'    # split_seg 用
        _iter2_cls_label = 'Refined Classification'
        _iter2_rsn_label = 'Refined Reasoning'    # split_seg 用
    else:
        _iter1_cls_label = 'Expert Classification'
        _iter1_rsn_label = 'Expert Reasoning'     # split_seg 用
        _iter2_cls_label = None                   # sem=False 无 iter2
        _iter2_rsn_label = None

    # ── 加载 iter1（channel pkls）────────────────────────────────────────
    # sem=True  + _show_iter1=False → 跳过，节省 I/O
    # sem=False                     → 始终加载（noguide 是唯一来源）
    iter1_reason = [None] * n
    iter1_result = [None] * n

    if _show_iter1:
        existing = [p for p in channel_pkls if os.path.exists(p)]
        missing  = [p for p in channel_pkls if not os.path.exists(p)]
        if missing:
            print(f'[load_product] 警告: {len(missing)} 个 channel pkl 不存在，已跳过:')
            for p in missing:
                print(f'  {p}')
        if existing:
            lbl = 'Expert' if not sem else 'Initial'
            print(f'[load_product] 构建{lbl}推理（{len(existing)} 个 channel pkl）...')
            for pkl_path in existing:
                reasons, results = _pkl_split_by_lcc(pkl_path, lcc_global_ids)
                for i, (r, s) in enumerate(zip(reasons, results)):
                    if iter1_reason[i] is None and (r is not None or s is not None):
                        iter1_reason[i] = r
                        iter1_result[i] = s
            n_covered = sum(1 for r in iter1_result if r is not None)
            print(f'[load_product] {lbl}推理覆盖 {n_covered}/{n} 个节点')
        else:
            print(f'[load_product] 警告: 无可用 channel pkl，'
                  f'{_iter1_cls_label} 将为空。')
    else:
        # sem=True + include_iter1=False：跳过 iter1 以节省 I/O
        print(f'[load_product] include_iter1=False：跳过 iter1 channel pkl 加载。')

    # ── 加载 iter2（仅 sem=True）─────────────────────────────────────────
    iter2_reason = [None] * n
    iter2_result = [None] * n

    if sem:
        if iter2_pkl and os.path.exists(iter2_pkl):
            print(f'[load_product] 加载 iter2 pkl: {iter2_pkl}')
            reasons, results = _pkl_split_by_lcc(iter2_pkl, lcc_global_ids)
            iter2_reason, iter2_result = reasons, results
            n_iter2 = sum(1 for r in iter2_result if r is not None)
            print(f'[load_product] iter2 推理覆盖 {n_iter2}/{n} 个节点')
        else:
            print('[load_product] 警告: iter2 pkl 不存在或未指定，'
                  'Refined Classification 将为空。')
    # sem=False: iter2 始终为空（noguide 来源在 iter1_* 数组中）

    # ── 辅助：字符级截断 ──────────────────────────────────────────────────
    def _maybe_trunc(text, max_chars):
        """max_chars<=0 时不截断。"""
        if not text or not max_chars:
            return text
        return text[:max_chars] + ('...' if len(text) > max_chars else '')

    # ── title 截断（防御性，仅应对极端异常值）────────────────────────────
    # 实测分布：avg=162t, p95=448t, p99=842t, max=48972t (0.2% 异常)
    # title_trunc_chars=0 时完全不截断（result_first 格式推荐，title 是分类信号）
    # title_trunc_chars=2000 时可防御极端节点 OOM（split_seg 格式可选）
    _title_trunc = title_trunc_chars or 0
    if _title_trunc:
        n_trunc = sum(1 for t in text_list if len(t) > _title_trunc)
        if n_trunc:
            print(f'[load_product] title_trunc_chars={_title_trunc}：截断 {n_trunc}/{len(text_list)} 个过长节点 '
                  f'({n_trunc/len(text_list)*100:.2f}%)')
        _effective_text_list = [_maybe_trunc(t, _title_trunc) for t in text_list]
    else:
        _effective_text_list = text_list

    # ── 拼装最终文本（text_fmt 只影响结构，不影响内容范围）──────────────
    text2 = []

    if text_fmt == 'result_first':
        # 结论前置：Classification: {result} \n Reasoning: {reason}
        for i in range(n):
            content = f'Title and Abstract:\n{_effective_text_list[i]}'

            # iter1 / Expert（sem=False 始终展示；sem=True 受 _show_iter1 控制）
            s1, r1 = iter1_result[i], iter1_reason[i]
            if s1 is not None or r1 is not None:
                content += f'\n\n{_iter1_cls_label}:'
                if s1 is not None:
                    content += f' {s1}'
                if r1 is not None:
                    content += f'\nReasoning: {_maybe_trunc(str(r1), reason_trunc_chars)}'

            # iter2 / Refined（仅 sem=True）
            s2, r2 = iter2_result[i], iter2_reason[i]
            if s2 is not None or r2 is not None:
                content += f'\n\n{_iter2_cls_label}:'
                if s2 is not None:
                    content += f' {s2}'
                if r2 is not None:
                    content += f'\nReasoning: {_maybe_trunc(str(r2), reason_trunc_chars)}'

            text2.append(content)

    elif text_fmt == 'reason_first':
        # 推理在前，结论在后（旧版格式）
        for i in range(n):
            content = f'Title and Abstract:\n{_effective_text_list[i]}'

            s1, r1 = iter1_result[i], iter1_reason[i]
            if s1 is not None or r1 is not None:
                content += f'\n\n{_iter1_cls_label} and Reasoning:'
                if r1 is not None:
                    content += f'\n{_maybe_trunc(str(r1), reason_trunc_chars)}'
                if s1 is not None:
                    content += f'\nClassification: {s1}'

            s2, r2 = iter2_result[i], iter2_reason[i]
            if s2 is not None or r2 is not None:
                content += f'\n\n{_iter2_cls_label} and Reasoning:'
                if r2 is not None:
                    content += f'\n{_maybe_trunc(str(r2), reason_trunc_chars)}'
                if s2 is not None:
                    content += f'\nClassification: {s2}'

            text2.append(content)

    else:  # 'split_seg'
        # ── 强制分离：result / reason / iter 各占独立 chunk ──────────────
        #
        # merge_cls_reason=False（默认，原始行为）：
        #   每段布局（sem=True, include_iter1=True，最多 5 段）：
        #     Seg0: Title and Abstract              ← chunk 0
        #     Seg1: Initial Classification: {s1}   ← chunk 1（iter1 结论）
        #     Seg2: Initial Reasoning: {r1}         ← chunk 2（iter1 推理）
        #     Seg3: Refined Classification: {s2}   ← chunk 3（iter2 结论）
        #     Seg4: Refined Reasoning: {r2}         ← chunk 4（iter2 推理）
        #
        # merge_cls_reason=True（方案B）：结论 + 推理合并为同一段，避免退化短 chunk：
        #   sem=True, include_iter1=True（最多 3 段）：
        #     Seg0: Title and Abstract
        #     Seg1: Initial Classification: {s1}\n\nInitial Reasoning:\n{r1}
        #     Seg2: Refined Classification: {s2}\n\nRefined Reasoning:\n{r2}
        #
        #   sem=True, include_iter1=False（最多 2 段）：
        #     Seg0: Title and Abstract
        #     Seg1: Refined Classification: {s2}\n\nRefined Reasoning:\n{r2}
        #
        #   sem=False（最多 2 段）：
        #     Seg0: Title and Abstract
        #     Seg1: Expert Classification: {s1}\n\nExpert Reasoning:\n{r1}
        for i in range(n):
            segs = [f'Title and Abstract:\n{_effective_text_list[i]}']

            s1, r1 = iter1_result[i], iter1_reason[i]
            s2, r2 = iter2_result[i], iter2_reason[i]

            if merge_cls_reason:
                # ── 方案B：结论+推理合并入同一段 ──────────────────────────
                # iter1 / Expert
                if s1 is not None or r1 is not None:
                    seg = ''
                    if s1 is not None:
                        seg += f'{_iter1_cls_label}: {s1}'
                    if r1 is not None:
                        seg += f'\n\n{_iter1_rsn_label}:\n{_maybe_trunc(str(r1), reason_trunc_chars)}'
                    segs.append(seg.strip())

                # iter2 / Refined（仅 sem=True）
                if s2 is not None or r2 is not None:
                    seg = ''
                    if s2 is not None:
                        seg += f'{_iter2_cls_label}: {s2}'
                    if r2 is not None:
                        seg += f'\n\n{_iter2_rsn_label}:\n{_maybe_trunc(str(r2), reason_trunc_chars)}'
                    segs.append(seg.strip())
            else:
                # ── 原始行为：结论/推理各占独立段 ──────────────────────────
                # iter1 / Expert
                if s1 is not None:
                    segs.append(f'{_iter1_cls_label}: {s1}')
                if r1 is not None:
                    segs.append(
                        f'{_iter1_rsn_label}:\n{_maybe_trunc(str(r1), reason_trunc_chars)}'
                    )

                # iter2 / Refined（仅 sem=True）
                if s2 is not None:
                    segs.append(f'{_iter2_cls_label}: {s2}')
                if r2 is not None:
                    segs.append(
                        f'{_iter2_rsn_label}:\n{_maybe_trunc(str(r2), reason_trunc_chars)}'
                    )

            text2.append(SEG_SEP.join(segs))

        seg_counts = [t.count(SEG_SEP) + 1 for t in text2]
        n_segs_avg = sum(seg_counts) / max(len(text2), 1)
        n_segs_max = max(seg_counts) if seg_counts else 0
        _merge_tag = '（方案B: cls+reason合并）' if merge_cls_reason else '（结论/推理独立）'
        print(f'[load_product] split_seg{_merge_tag}: '
              f'平均 {n_segs_avg:.1f} 段/节点，最多 {n_segs_max} 段，'
              f'总节点数 {len(text2)}')

    # ── 提取 LLM 预测标签并写入 data.llm_label_idx ──────────────────────────
    # 以"主要来源"的 result 字段做提取：sem=True→iter2_result，sem=False→iter1_result
    # 对无 pkl 覆盖的节点（result=None）→ 返回 -1
    _primary_result = iter2_result if sem else iter1_result
    llm_label_idx = torch.full((n,), -1, dtype=torch.long)
    n_matched = 0
    for i, res in enumerate(_primary_result):
        if res is not None:
            idx = _extract_llm_label_idx(str(res))
            llm_label_idx[i] = idx
            if idx >= 0:
                n_matched += 1

    # 将全局 LCC 节点数组扩展到完整 num_nodes（未覆盖节点保持 -1）
    # data.num_nodes 已在 _compact_to_lcc 后等于 LCC 节点数，此处直接赋值
    data.llm_label_idx = llm_label_idx
    print(f'[load_product] llm_label_idx: {n_matched}/{n} 个节点成功匹配类别索引')

    return text2
