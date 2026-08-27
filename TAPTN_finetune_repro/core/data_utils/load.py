import os
import re
import json
import pickle
import torch
import csv
from core.data_utils.dataset import CustomDGLDataset


# ── GPT 标签提取辅助（用于 label_prefix_in_text 方案B）─────────────────────────
# cora / arxiv_2023 的 GPT 回应中类别别名映射
_CORA_ALIASES = {
    'case based': 'Case Based',
    'case-based': 'Case Based',
    'genetic algorithms': 'Genetic Algorithms',
    'neural networks': 'Neural Networks',
    'probabilistic methods': 'Probabilistic Methods',
    'probabilistic method': 'Probabilistic Methods',
    'reinforcement learning': 'Reinforcement Learning',
    'rule learning': 'Rule Learning',
    # "Theory" 统一写成 Computational Learning Theory
    'theory': 'Computational Learning Theory',
    'computational learning theory': 'Computational Learning Theory',
    'computational learning': 'Computational Learning Theory',
    'learning theory': 'Computational Learning Theory',
    'cl theory': 'Computational Learning Theory',
}

_SCORE_PAT = re.compile(
    r'\*{0,2}([\w ][\w ,/-]*?)\*{0,2}\s*\((?:Relevance\s+)?Score[:\s]*([0-9.]+)\)',
    re.IGNORECASE,
)


def _match_cora_label(name: str):
    """将 GPT 输出的类别名（容忍大小写/连字符/下划线）映射到规范名称，失败返回 None。"""
    key = re.sub(r'[\-_]+', ' ', name.lower()).strip()
    if key in _CORA_ALIASES:
        return _CORA_ALIASES[key]
    for alias, canon in _CORA_ALIASES.items():
        if alias in key or key in alias:
            return canon
    return None


def _extract_gpt_label(text: str):
    """从 GPT 精炼回应文本中提取置信度最高的标签规范名，提取失败返回 None。"""
    # 方法1: 找 (Relevance Score: x) 格式，取分值最高的
    hits = _SCORE_PAT.findall(text)
    if hits:
        scored = []
        for name, score_s in hits:
            canon = _match_cora_label(name.strip())
            if canon is not None:
                try:
                    scored.append((float(score_s), canon))
                except ValueError:
                    pass
        if scored:
            return max(scored, key=lambda x: x[0])[1]
    # 方法2: 加粗文本 **Label**
    for m in re.finditer(r'\*\*([\w][\w ,/-]+?)\*\*', text):
        canon = _match_cora_label(m.group(1).strip())
        if canon is not None:
            return canon
    # 方法3: 全文扫描（兜底，精度较低）
    for alias in sorted(_CORA_ALIASES, key=len, reverse=True):
        if alias in text.lower():
            return _CORA_ALIASES[alias]
    return None


# cora 规范标签名 → 类别索引（与 parse_cora() 中 class_map 顺序一致）
_CORA_LABEL_TO_IDX = {
    'Case Based': 0,
    'Genetic Algorithms': 1,
    'Neural Networks': 2,
    'Probabilistic Methods': 3,
    'Reinforcement Learning': 4,
    'Rule Learning': 5,
    'Computational Learning Theory': 6,
}


def _extract_gpt_soft_label(text: str, n_classes: int = 7):
    """从 GPT 精炼回应中提取 top-2 预测标签及其评分，返回归一化软分布张量。

    返回 FloatTensor(n_classes,)：
    - 若找到带分值的预测：对所有有效（标签已知）的 (score, label) 对取 top-2，
      按分值归一化（softmax-like）构成软独热向量。
    - 若只找到加粗标签无分值：top-1 位置为 1.0，其余为 0。
    - 提取失败：全零向量（lm_trainer 中忽略全零行不计入损失）。
    """
    import torch as _torch
    vec = _torch.zeros(n_classes, dtype=_torch.float32)

    # ── 方法1：有分值的格式 ────────────────────────────────────────────────
    hits = _SCORE_PAT.findall(text)
    if hits:
        scored = []
        for name, score_s in hits:
            canon = _match_cora_label(name.strip())
            if canon is not None and canon in _CORA_LABEL_TO_IDX:
                try:
                    scored.append((float(score_s), _CORA_LABEL_TO_IDX[canon]))
                except ValueError:
                    pass
        if scored:
            # 去重：同一类别保留最高分，再取 top-2
            best = {}
            for sc, idx in scored:
                if idx not in best or sc > best[idx]:
                    best[idx] = sc
            top2 = sorted(best.items(), key=lambda x: -x[1])[:2]
            total = sum(sc for _, sc in top2)
            if total > 0:
                for idx, sc in top2:
                    vec[idx] = sc / total
            return vec

    # ── 方法2：加粗标签无分值 ─────────────────────────────────────────────
    for m in re.finditer(r'\*\*([\w][\w ,/-]+?)\*\*', text):
        canon = _match_cora_label(m.group(1).strip())
        if canon is not None and canon in _CORA_LABEL_TO_IDX:
            vec[_CORA_LABEL_TO_IDX[canon]] = 1.0
            return vec

    return vec   # 全零：提取失败


def load_gpt_preds(dataset, topk):
    using_pkl=False
    if using_pkl:
        import pickle
        fn = f'gpt_preds/{dataset}.pkl'
        print(f"Loading topk preds from {fn}")
        return pickle.load(open(fn, 'rb'))
    preds = []
    fn = f'gpt_preds/{dataset}.csv'
    print(f"Loading topk preds from {fn}")
    with open(fn, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            inner_list = []
            for value in row:
                inner_list.append(int(value))
            preds.append(inner_list)

    pl = torch.zeros(len(preds), topk, dtype=torch.long)
    for i, pred in enumerate(preds):
        pl[i][:len(pred)] = torch.tensor(pred[:topk], dtype=torch.long)+1
    return pl


def load_data(dataset, use_dgl=False, use_text=False, use_gpt=False, seed=0, cfg=None, sem=True):
    if dataset == 'cora':
        from core.data_utils.load_cora import get_raw_text_cora as get_raw_text
        num_classes = 7
    elif dataset == 'arxiv_2023':
        from core.data_utils.load_arxiv_2023 import get_raw_text_arxiv_2023 as get_raw_text
        num_classes = 40
    elif dataset in ['wisconsin', 'texas', 'cornell']:
        from core.data_utils.load_wisconsin import load_wisconsin as get_raw_text
        num_classes = 7
    elif dataset == 'actor':
        from core.data_utils.load_actor import (
            get_raw_text_actor, build_actor_gpt_text,
            DEFAULT_CHANNEL_PKLS, DEFAULT_ITER2_PKL,
            DEFAULT_CHANNELS_NODE_SET, DEFAULT_SHOW_CONSENSUS,
        )
        num_classes = 5
    elif dataset == 'product':
        import core.data_utils.load_product as _lp_mod
        get_raw_text_product    = _lp_mod.get_raw_text_product
        build_product_gpt_text  = _lp_mod.build_product_gpt_text
        PRODUCT_DEFAULT_CHANNEL_PKLS        = _lp_mod.DEFAULT_CHANNEL_PKLS
        PRODUCT_DEFAULT_ITER2_PKL           = _lp_mod.DEFAULT_ITER2_PKL
        PRODUCT_DEFAULT_NOSEM_CHANNEL_PKLS  = _lp_mod.DEFAULT_NOSEM_CHANNEL_PKLS
        PRODUCT_DEFAULT_NOSEM_ITER2_PKL     = _lp_mod.DEFAULT_NOSEM_ITER2_PKL
        PRODUCT_DEFAULT_CHANNELS_NODE_SET   = _lp_mod.DEFAULT_CHANNELS_NODE_SET
        PRODUCT_DEFAULT_SHOW_CONSENSUS      = _lp_mod.DEFAULT_SHOW_CONSENSUS
        num_classes = 47
    else:
        exit(f'Error: Dataset {dataset} not supported')

    # ── for training GNN (no text needed) ──────────────────────────────
    if not use_text:
        if dataset == 'actor':
            data, _ = get_raw_text_actor(use_text=False, seed=seed)
        elif dataset == 'product':
            data, _ = get_raw_text_product(use_text=False, seed=seed)
        elif dataset in ['wisconsin', 'texas', 'cornell']:
            data, _ = get_raw_text(use_text=False, seed=seed, data_set_name=dataset)
        else:
            data, _ = get_raw_text(use_text=False, seed=seed)
        if use_dgl:
            data = CustomDGLDataset(dataset, data)
        return data, num_classes

    # ── for finetuning LM ───────────────────────────────────────────────
    if use_gpt:
        # ── Actor: multi-channel initial + iter2 refined ──────────────
        if dataset == 'actor':
            # 读取 cfg 中的 Actor 参数（若无 cfg 则使用默认值）
            _ch_pkls   = (getattr(cfg.actor, 'channel_pkls',   DEFAULT_CHANNEL_PKLS)
                          if cfg is not None and hasattr(cfg, 'actor')
                          else DEFAULT_CHANNEL_PKLS)
            _iter2_pkl = (getattr(cfg.actor, 'iter2_pkl',      DEFAULT_ITER2_PKL)
                          if cfg is not None and hasattr(cfg, 'actor')
                          else DEFAULT_ITER2_PKL)
            _ch_ns     = (getattr(cfg.actor, 'channels_node_set', DEFAULT_CHANNELS_NODE_SET)
                          if cfg is not None and hasattr(cfg, 'actor')
                          else DEFAULT_CHANNELS_NODE_SET)
            _show_con  = (getattr(cfg.actor, 'show_consensus',  DEFAULT_SHOW_CONSENSUS)
                          if cfg is not None and hasattr(cfg, 'actor')
                          else DEFAULT_SHOW_CONSENSUS)

            # get_raw_text_actor 返回 (data, text_list)；data 已完成 test 集重切分
            data, text_list = get_raw_text_actor(use_text=True, seed=seed)

            # build_actor_gpt_text 内部会自动获取 text_dict（如未传入）
            text = build_actor_gpt_text(
                data, text_list,
                channel_pkls=list(_ch_pkls),
                iter2_pkl=_iter2_pkl,
                channels_node_set=_ch_ns,
                show_consensus=_show_con,
                seed=seed,
            )

        # ── Product: single/multi-channel initial + optional iter2 ──
        elif dataset == 'product':
            _has_prod_cfg = cfg is not None and hasattr(cfg, 'product')
            if sem:
                # 结构引导模式（guided pkls）
                _ch_pkls   = (getattr(cfg.product, 'channel_pkls',      PRODUCT_DEFAULT_CHANNEL_PKLS)
                              if _has_prod_cfg else PRODUCT_DEFAULT_CHANNEL_PKLS)
                _iter2_pkl = (getattr(cfg.product, 'iter2_pkl',         PRODUCT_DEFAULT_ITER2_PKL)
                              if _has_prod_cfg else PRODUCT_DEFAULT_ITER2_PKL)
            else:
                # 无结构模式（noguide pkl，无 iter2）
                _ch_pkls   = (getattr(cfg.product, 'nosem_channel_pkls', PRODUCT_DEFAULT_NOSEM_CHANNEL_PKLS)
                              if _has_prod_cfg else PRODUCT_DEFAULT_NOSEM_CHANNEL_PKLS)
                _iter2_pkl = (getattr(cfg.product, 'nosem_iter2_pkl',    PRODUCT_DEFAULT_NOSEM_ITER2_PKL)
                              if _has_prod_cfg else PRODUCT_DEFAULT_NOSEM_ITER2_PKL)
            _ch_ns    = (getattr(cfg.product, 'channels_node_set', PRODUCT_DEFAULT_CHANNELS_NODE_SET)
                         if _has_prod_cfg else PRODUCT_DEFAULT_CHANNELS_NODE_SET)
            _show_con = (getattr(cfg.product, 'show_consensus',    PRODUCT_DEFAULT_SHOW_CONSENSUS)
                         if _has_prod_cfg else PRODUCT_DEFAULT_SHOW_CONSENSUS)

            _reason_trunc       = (getattr(cfg.product, 'reason_trunc_chars', 0)
                                   if _has_prod_cfg else 0)
            _title_trunc        = (getattr(cfg.product, 'title_trunc_chars',  0)
                                   if _has_prod_cfg else 0)
            _text_fmt           = (getattr(cfg.product, 'text_fmt',            'result_first')
                                   if _has_prod_cfg else 'result_first')
            _include_iter1      = (getattr(cfg.product, 'include_iter1',       True)
                                   if _has_prod_cfg else True)
            _merge_cls_reason   = (getattr(cfg.product, 'split_seg_merge_cls_reason', False)
                                   if _has_prod_cfg else False)
            print(f'[load_data] product 模式: {"sem" if sem else "no_sem"}, '
                  f'text_fmt={_text_fmt}, include_iter1={_include_iter1}, '
                  f'reason_trunc_chars={_reason_trunc}, title_trunc_chars={_title_trunc}, '
                  f'merge_cls_reason={_merge_cls_reason}')

            # get_raw_text_product 返回 (data, text_list)；data 已完成两阶段切分
            data, text_list = get_raw_text_product(use_text=True, seed=seed)

            # build_product_gpt_text 内部会自动获取 text_dict（如未传入）
            # 同时会在 data 上写入 data.llm_label_idx（供 lm_trainer 使用）
            text = build_product_gpt_text(
                data, text_list,
                channel_pkls=list(_ch_pkls),
                iter2_pkl=_iter2_pkl,
                channels_node_set=_ch_ns,
                show_consensus=_show_con,
                sem=sem,
                seed=seed,
                reason_trunc_chars=_reason_trunc,
                title_trunc_chars=_title_trunc,
                text_fmt=_text_fmt,
                include_iter1=_include_iter1,
                merge_cls_reason=_merge_cls_reason,
            )

        # ── WebKB ─────────────────────────────────────────────────────
        elif dataset in ['wisconsin', 'texas', 'cornell']:
            data, text = get_raw_text(use_text=True, seed=seed, data_set_name=dataset)
            folder_path = 'gpt_responses/{}'.format(dataset)
            print(f"using gpt: {folder_path}")
            n = data.y.shape[0]
            text2 = []
            def _load_webkb_pkl(name):
                assets = os.environ.get('TAPTN_ASSETS', '')
                for p in (name,
                          os.path.join(assets, name) if assets else '',
                          os.path.join(assets, 'pkls', name) if assets else ''):
                    if p and os.path.isfile(p):
                        with open(p, 'rb') as f:
                            return pickle.load(f)
                with open(name, 'rb') as f:
                    return pickle.load(f)
            initial_reason = _load_webkb_pkl(f'{dataset}2_hop1_guide.pkl')
            refined_reason  = _load_webkb_pkl(f'{dataset}2_hop1_guide_2.pkl')
            valid_id  = torch.where(data.y != 6)[0].cpu().numpy()
            valid_map = {valid_id[i]: i for i in range(len(valid_id))}
            for i in range(n):
                content = f'Title and Abstract:\n{text[i]}'
                content += '\n\nInitial Classification and Rasoning:\n{}'.format(
                    initial_reason['reason'][i])
                if i in valid_map:
                    content += '\n\nRefined Classification and Rasoning:\n{}'.format(
                        refined_reason['reason'][valid_map[i]])
                text2.append(content)
            text = text2

        # ── Cora / arxiv_2023 ────────────────────────────────────────
        # 各数据集的 GPT 响应目录：
        #   sem=True  → initial: {dataset}_sem3/  refined: {dataset}/
        #   sem=False → initial: {dataset}_nosem2/ refined: {dataset}/
        #
        # 方案A/B/C 参数仅对 cora 有效，其他数据集（如 arxiv_2023）强制忽略。
        else:
            data, text = get_raw_text(use_text=True, seed=seed)
            folder_path = 'gpt_responses/{}'.format(dataset)
            initial_suffix = '_sem3' if sem else '_nosem2'

            # ── include_initial / label_prefix / soft_label 逻辑 ────────────────
            _lm_train = (getattr(getattr(cfg, 'lm', None), 'train', None)
                         if cfg is not None else None)
            _is_cora = (dataset == 'cora')

            # include_initial_reasoning（方案A控制开关）：
            #   · cora        → 由参数控制（默认 False = 只用 refined，性能更好）
            #   · 其他数据集   → 永远 True，与修改前原始行为完全一致（双iter）
            if _is_cora:
                _include_initial = (
                    _lm_train is not None
                    and bool(getattr(_lm_train, 'include_initial_reasoning', False))
                )
            else:
                _include_initial = True   # 原始行为：双iter，不受参数影响

            # 方案B/C 严格限定 cora-only，其余数据集强制 False
            _label_prefix = (
                _is_cora and _lm_train is not None
                and bool(getattr(_lm_train, 'label_prefix_in_text', False))
            )
            _soft_label = (
                _is_cora and _lm_train is not None
                and bool(getattr(_lm_train, 'use_gpt_soft_label', False))
            )

            print(f"using gpt ({'sem' if sem else 'nosem'}, dataset={dataset}): "
                  f"{'initial=' + folder_path + initial_suffix + '  ' if _include_initial else ''}"
                  f"refined={folder_path}  "
                  f"include_initial={_include_initial}  "
                  f"label_prefix={_label_prefix}  "
                  f"soft_label={_soft_label}")
            n = data.y.shape[0]
            text2 = []
            _prefix_hit = _prefix_miss = 0

            # ── 方案C 软独热预存储 ─────────────────────────────────────────────
            # gpt_soft_label[i]: FloatTensor(n_classes,) — top-2 分值归一化后的软分布
            # 未能提取时该行全零（loss 计算时忽略全零行）
            if _soft_label:
                import torch as _torch
                _n_cls = 7   # cora 固定 7 类
                _soft_mat = _torch.zeros(n, _n_cls, dtype=_torch.float32)

            for i in range(n):
                filename = str(i) + '.json'
                # ── 读取 refined 响应（必读）──────────────────────────────────
                with open(os.path.join(folder_path, filename)) as _f:
                    refined_content = json.load(_f)['choices'][0]['message']['content']

                # ── 方案B：标签前置 ────────────────────────────────────────────
                if _label_prefix:
                    pred_label = _extract_gpt_label(refined_content)
                    if pred_label is not None:
                        _prefix_hit += 1
                        prefix = f'[Predicted Category: {pred_label}]\n\n'
                    else:
                        _prefix_miss += 1
                        prefix = ''
                else:
                    prefix = ''

                content = f'{prefix}Title and Abstract:\n{text[i]}'

                # ── 可选：拼接 initial 推理 ────────────────────────────────────
                if _include_initial:
                    with open(os.path.join(folder_path + initial_suffix, filename)) as _f:
                        initial_content = json.load(_f)['choices'][0]['message']['content']
                    content += '\n\nInitial Classification and Rasoning:\n{}'.format(initial_content)

                content += '\n\nRefined Classification and Rasoning:\n{}'.format(refined_content)
                text2.append(content)

                # ── 方案C：提取 top-2 软独热 ─────────────────────────────────
                if _soft_label:
                    _soft_mat[i] = _extract_gpt_soft_label(refined_content, _n_cls)

            if _label_prefix:
                print(f'[label_prefix] 提取成功: {_prefix_hit}/{n}  '
                      f'失败退化为无前缀: {_prefix_miss}/{n}')
            if _soft_label:
                _hit = (_soft_mat.sum(dim=1) > 0).sum().item()
                print(f'[gpt_soft_label] 提取成功: {_hit}/{n}  '
                      f'全零（无法提取）: {n - _hit}/{n}')
                data.gpt_soft_label = _soft_mat   # 挂载到 data 对象供 lm_trainer 使用
            text = text2

    # ── plain text (no GPT reasoning) ──────────────────────────────────
    else:
        if dataset == 'actor':
            data, text = get_raw_text_actor(use_text=True, seed=seed)
        elif dataset == 'product':
            data, text = get_raw_text_product(use_text=True, seed=seed)
        elif dataset in ['wisconsin', 'texas', 'cornell']:
            data, text = get_raw_text(use_text=True, seed=seed, data_set_name=dataset)
        else:
            data, text = get_raw_text(use_text=True, seed=seed)

    return data, num_classes, text
