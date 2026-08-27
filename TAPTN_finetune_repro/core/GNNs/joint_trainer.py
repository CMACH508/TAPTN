"""
core/GNNs/joint_trainer.py

P4 流水线：GNN 与 LM 联合训练（Joint Fine-tuning）。

与普通 GNNTrainer 的区别
─────────────────────────────────────────────────────────────────────
· 加载 P1/P2 产生的 LM checkpoint（prt_lm/{dataset}/{lm_model}-seed{seed}.ckpt）
  作为 LM 初始权重；若 checkpoint 不存在则使用预训练权重。
· 训练每步对 train_mask 节点用 LM 动态计算 embedding（梯度流经 LM）。
· 其余节点使用缓存 embedding（每 LM_REFRESH_INTERVAL 轮刷新一次）。
· 评估（early stopping）使用当前缓存，快速且轻量。
· eval_and_save() 做最终全图 LM 推理后报告精度。
· 训练结束后只保存 GNN checkpoint，不保存 LM。
· 不支持 RevGAT（RevGAT 依赖 DGL），遇到时会抛出 ValueError。
· 显存优化：fp16 autocast + GradScaler + 小 batch，避免 DeBERTa 的
  disentangled attention 在 fp32 下峰值超过 30 GB。
─────────────────────────────────────────────────────────────────────
"""

import os
import torch
import torch.nn as nn
import numpy as np
from time import time
from contextlib import contextmanager

from transformers import AutoTokenizer, AutoModel
from core.LMs.model import BertClassifier
from core.GNNs.gnn_utils import EarlyStopping, Evaluator
from core.data_utils.load import load_data
from core.utils import time_logger, init_path, lm_pretrained_path, lm_model_root, lm_ckpt_stem, lm_emb_path, infer_emb_dim

LOG_FREQ = 10


class JointGNNLMTrainer:
    """P4: 在 GNN 训练阶段同时微调 LM，训练结束后不保存 LM。"""

    # LM_BATCH_SIZE 仅作类级别默认回退值；
    # 实际使用 self._lm_batch_size（由 _init_lm_batch_size 动态确定）。
    LM_BATCH_SIZE: int = 32

    # ── 各数据集的 tokenizer max_length 默认值 ──────────────────────────────
    # cfg.lm.train.p4_max_length > 0 时强制使用指定值（覆盖此表）。
    # cora      : 中位数 180 tok，93% 节点 ≤256 tok，设 256 仅截断 7%
    # arxiv_2023: 中位数 264 tok，P90=371，P95=405，设 384 仅截断 7.3%
    #             DeBERTa disentangled attention 显存/计算与 seq_len² 成正比；
    #             512→384：(384/512)²≈0.56，节省 ~44% LM 计算，对精度影响极小。
    #             【修复】原字典写了 'arxiv': 256，但 dataset_name='arxiv_2023'，
    #             导致 key miss → 回退 512，P4 训练耗时 1.3h+。
    # 其余数据集: 保持 512（原始行为）。
    DATASET_MAX_LENGTH: dict = {
        'cora':       256,
        'arxiv_2023': 384,
    }
    _DEFAULT_MAX_LENGTH: int = 512  # 未在 DATASET_MAX_LENGTH 中列出时的回退值

    # ── 各数据集的 LM embedding 缓存刷新间隔（epoch 数） ─────────────────────
    # cfg.lm.train.p4_refresh_interval > 0 时强制使用指定值（覆盖此表）。
    # 小数据集（cora 2.7K 节点）：间隔 10 epoch，刷新代价低（<5s/次）。
    # 大数据集（arxiv_2023 33.8K 节点）：间隔 20 epoch，每次刷新耗时 ~155s；
    #   间隔翻倍（10→20）减少约 6 次刷新 ≈ 930s，且对最终精度无影响
    #   （eval_and_save 始终用最新 LM 权重重推理）。
    DATASET_REFRESH_INTERVAL: dict = {
        'arxiv_2023': 20,
    }
    _DEFAULT_REFRESH_INTERVAL: int = 10  # 未在 DATASET_REFRESH_INTERVAL 中列出时的回退值

    def __init__(self, cfg, feature_type: str = "TA"):
        if cfg.gnn.model.name == "RevGAT":
            raise ValueError(
                "[P4] JointGNNLMTrainer 不支持 RevGAT（RevGAT 依赖 DGL）。"
                "P4 流水线会自动跳过 RevGAT。"
            )

        self.seed          = cfg.seed
        self.device_id     = cfg.device
        self.device        = torch.device(
            f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu"
        )
        self.dataset_name  = cfg.dataset
        self.gnn_model_name = cfg.gnn.model.name
        self.lm_model_name  = cfg.lm.model.name
        self.hidden_dim    = cfg.gnn.model.hidden_dim
        self.num_layers    = cfg.gnn.model.num_layers
        self.dropout       = cfg.gnn.train.dropout
        self.gnn_lr        = cfg.gnn.train.lr
        self.lm_lr         = cfg.lm.train.lr
        self.lm_wd         = cfg.lm.train.weight_decay
        self.feature_type  = feature_type
        self.epochs        = cfg.gnn.train.epochs

        # ── P4 是否使用 GPT 增强文本（默认 False，用原始属性文本）──────────────
        # 设为 True 时：加载 GPT 增强文本（与 P2 一致），使用 use_gpt=True 的 LM
        # 对应 config.py: cfg.lm.train.p4_use_gpt = True
        self.p4_use_gpt = bool(getattr(cfg.lm.train, 'p4_use_gpt', False))

        # ── fp16 autocast（默认启用，大幅降低 DeBERTa 显存需求）─────────────
        # 实际批次大小由 _init_lm_batch_size() 在模型加载后动态确定；
        # 三层策略：cfg.p4_lm_batch_size(>0) > GPU显存估算 > OOM自动减半兜底
        self.use_amp = torch.cuda.is_available()
        self.scaler  = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        # 暂存配置中的批次偏好，LM 加载后再用 _init_lm_batch_size() 计算实际值
        self._cfg_lm_batch_size = int(
            getattr(cfg.lm.train, 'p4_lm_batch_size', 0)
        )
        self._lm_batch_size = self.LM_BATCH_SIZE  # 临时默认，_init_lm_batch_size 会覆盖
        # tokenizer max_length：cfg=0 时按 DATASET_MAX_LENGTH 数据集自动选取，
        # cfg>0 时强制使用指定值（覆盖数据集默认）。
        _cfg_max = int(getattr(cfg.lm.train, 'p4_max_length', 0))
        if _cfg_max > 0:
            self._p4_max_length = _cfg_max
        else:
            self._p4_max_length = self.DATASET_MAX_LENGTH.get(
                self.dataset_name, self._DEFAULT_MAX_LENGTH
            )

        # LM refresh 间隔：cfg=0 时按 DATASET_REFRESH_INTERVAL 自动选取，
        # cfg>0 时强制使用指定值。
        _cfg_ri = int(getattr(cfg.lm.train, 'p4_refresh_interval', 0))
        if _cfg_ri > 0:
            self._lm_refresh_interval = _cfg_ri
        else:
            self._lm_refresh_interval = self.DATASET_REFRESH_INTERVAL.get(
                self.dataset_name, self._DEFAULT_REFRESH_INTERVAL
            )

        # ── 加载图结构数据（不需要文本） ─────────────────────────────────────
        data, num_classes = load_data(
            self.dataset_name, use_dgl=False, use_text=False, seed=self.seed
        )
        self.num_nodes  = data.y.shape[0]
        self.num_classes = num_classes
        data.y = data.y.squeeze()
        self.data = data.to(self.device)

        # ── 初始化 LM（含 tokenize 全图文本） ────────────────────────────────
        self._init_lm(cfg)

        # ── 动态确定 LM 批次大小（模型加载后才能估算显存）────────────────────
        self._init_lm_batch_size()

        # ── 初始化 embedding 缓存（优先读 P1 .emb；否则 LM 全图推理）────────
        self._init_embedding_cache(cfg)

        # ── 初始化 GNN ────────────────────────────────────────────────────────
        self._init_gnn(num_classes)

        # ── 优化器 ────────────────────────────────────────────────────────────
        self.gnn_optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.gnn_lr, weight_decay=0.0
        )
        self.lm_optimizer = torch.optim.AdamW(
            self.lm_model.parameters(), lr=self.lm_lr, weight_decay=self.lm_wd
        )

        # ── Early Stopping & 评估工具 ─────────────────────────────────────────
        self.loss_func = nn.CrossEntropyLoss()
        self.ckpt      = f"output/{self.dataset_name}/{self.gnn_model_name}_joint.pt"
        self.stopper   = (
            EarlyStopping(patience=cfg.gnn.train.early_stop, path=self.ckpt)
            if cfg.gnn.train.early_stop > 0
            else None
        )

        self._evaluator = Evaluator(name=self.dataset_name)
        self.evaluator  = lambda pred, labels: self._evaluator.eval(
            {"y_pred": pred.argmax(dim=-1, keepdim=True),
             "y_true": labels.view(-1, 1)}
        )["acc"]

    # ──────────────────────────────────────────────────────────────────────────
    # 初始化辅助函数
    # ──────────────────────────────────────────────────────────────────────────

    def _init_lm(self, cfg):
        """加载 LM 并 tokenize 全图文本（存 CPU，按需移到 GPU）。

        p4_use_gpt=False（默认）：使用原始属性文本，兼容 P1 checkpoint。
        p4_use_gpt=True：使用 GPT 增强文本（与 P2 一致），use_gpt=True 的
                          BertClassifier（含 chunk 分割逻辑），加载 P2 checkpoint。
                          注意：P4 的 LM forward 目前仍只取首 512 token 的 CLS，
                          即使 use_gpt=True 也不做 sliding-window 分块（梯度开销过大）。
        """
        lm_root   = lm_model_root(cfg)
        lm_path   = lm_pretrained_path(self.lm_model_name, lm_root)
        tokenizer = AutoTokenizer.from_pretrained(lm_path)
        bert_model = AutoModel.from_pretrained(lm_path)

        # P4 始终以 use_gpt=False 创建 BertClassifier（不做分块推理，减少训练显存）
        # 即使文本来自 GPT 增强版本，P4 也只取首 512 token 的 CLS embedding
        self.lm_model = BertClassifier(
            bert_model,
            n_labels=self.num_classes,
            feat_shrink=cfg.lm.model.feat_shrink,
            use_gpt=False,  # P4 训练阶段始终用单段 CLS，避免 chunk 梯度显存爆炸
        ).to(self.device)

        # 尝试加载 checkpoint（P1 或 P2 均可；使用 strict=False 跳过架构不兼容层）
        lm_ckpt = f'{lm_ckpt_stem(cfg, self.dataset_name, "", self.seed)}.ckpt'
        if os.path.exists(lm_ckpt):
            print(f"[P4] 加载 LM checkpoint: {lm_ckpt}")
            state = torch.load(lm_ckpt, map_location=self.device, weights_only=False)
            missing, unexpected = self.lm_model.load_state_dict(state, strict=False)
            if missing:
                print(f"[P4]   缺少键（已忽略）: {missing}")
            if unexpected:
                print(f"[P4]   多余键（已忽略）: {unexpected}")
        else:
            print(f"[P4] 未找到 LM checkpoint ({lm_ckpt})，使用预训练权重初始化 LM")

        # ── gradient checkpointing：以重计算换显存 ────────────────────────────
        # 原理：前向时只保留各层边界 hidden state，反向时逐层重算 attention map
        # 效果：激活内存从 O(layers) 降至 O(1)，fp16 batch=8 峰值从 ~13 GB → ~2-3 GB
        # 代价：每次反向多一次前向，训练约慢 30%（P4 本就慢，可接受）
        self.lm_model.bert_encoder.gradient_checkpointing_enable()
        print("[P4] gradient checkpointing 已启用（激活显存 O(layers)→O(1)）")

        # 加载文本并 tokenize
        # p4_use_gpt=True → GPT 增强文本（result_first / split_seg 等格式）
        # p4_use_gpt=False → 原始属性文本
        _, _, text = load_data(
            self.dataset_name, use_dgl=False, use_text=True,
            use_gpt=self.p4_use_gpt, seed=self.seed
        )
        print(f"[P4] 文本来源: {'GPT 增强' if self.p4_use_gpt else '原始属性'}，"
              f"共 {len(text)} 条")

        # tokenize 全图文本（padding=True 对齐到本数据集最长序列，上限 p4_max_length）。
        # p4_max_length 默认 256（cfg.lm.train.p4_max_length）：
        #   · cora 文本中位数 180 tok，93% 节点自然 ≤256 tok，仅 7% 被截断
        #   · DeBERTa disentangled attention 的 p2c_att 张量 [batch×12, seq, 2×seq]
        #     与 seq_len² 成正比；256 vs 512 差 4× attention 显存
        #   · 对文本较长的数据集可通过 lm.train.p4_max_length 调大
        enc = tokenizer(
            text, padding=True, truncation=True, max_length=self._p4_max_length,
            return_tensors="pt"
        )
        print(f"[P4] tokenize max_length={self._p4_max_length}  "
              f"实际 pad 长度={enc['input_ids'].shape[1]}")
        # 存 CPU，推理时按批移到 GPU
        self.token_ids = {k: v.cpu() for k, v in enc.items()}

    def _init_embedding_cache(self, cfg):
        """
        优先从 P1 的 .emb 文件加载 embedding 缓存；
        若不存在则通过 LM 推理全图生成。
        """
        feat_shrink = cfg.lm.model.feat_shrink
        p1_emb = lm_emb_path(cfg, self.dataset_name, 'TA', self.seed)
        emb_dim = infer_emb_dim(p1_emb, self.num_nodes, feat_shrink)
        if os.path.exists(p1_emb):
            print(f"[P4] 从 P1 .emb 初始化 embedding 缓存: {p1_emb}")
            cache = torch.from_numpy(
                np.array(
                    np.memmap(p1_emb, mode="r", dtype=np.float16,
                              shape=(self.num_nodes, emb_dim))
                )
            ).to(torch.float32)
        else:
            print("[P4] 未找到 P1 .emb，通过 LM 推理全图初始化缓存...")
            cache = self._lm_infer_all()

        self.emb_cache = cache.to(self.device)   # [N, D]

    def _init_gnn(self, num_classes: int):
        """根据 cfg.gnn.model.name 实例化 GNN 模型。"""
        name    = self.gnn_model_name
        in_dim  = self.emb_cache.shape[1]
        h_dim   = self.hidden_dim
        n_cls   = num_classes
        n_layer = self.num_layers
        drop    = self.dropout

        if name == "SAGE":
            from core.GNNs.SAGE.model import SAGE as GNN
        elif name == "GCN2":
            from core.GNNs.GCN2.model import GCN as GNN
        elif name == "DirGNN":
            from core.GNNs.DirGNN.model import GAT as GNN
        elif name == "GAT":
            from core.GNNs.GATv2.model import GAT as GNN
        elif name in ("GATv2", "Saint"):
            from core.GNNs.GATv2.model import GATv2 as GNN
        elif name == "FSGNN":
            from core.GNNs.FSGNN.model import FSGNN as GNN
        elif name == "ACMGNN":
            from core.GNNs.ACMGNN.model import ACMGNN as GNN
        elif name == "DMP":
            from core.GNNs.DMP.model import DMP as GNN
        elif name == "APPNP":
            from core.GNNs.APPNP.model import APPNPModel as GNN
        elif name == "ChebNet":
            from core.GNNs.Cheb.model import ChebNet as GNN
        elif name == "ASC":
            from core.GNNs.ASC.model import ASC as GNN
        elif name == "GraphTARIF":
            from core.GNNs.GraphTARIF.model import GraphTARIF as GNN
        else:
            print(f"[P4] 模型 {name} 未识别，回退到 GATv2")
            from core.GNNs.GATv2.model import GATv2 as GNN

        self.model = GNN(
            in_channels=in_dim,
            hidden_channels=h_dim,
            out_channels=n_cls,
            num_layers=n_layer,
            dropout=drop,
            use_pred=False,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[P4] GNN 参数量: {n_params}")

    # ──────────────────────────────────────────────────────────────────────────
    # 动态批次大小：估算 + OOM 自动减半
    # ──────────────────────────────────────────────────────────────────────────

    def _estimate_lm_batch_size(self) -> int:
        """根据当前可用 GPU 显存估算安全的 LM 批次大小（fp16+grad_ckpt 经验公式）。

        在模型加载后调用，此时 mem_get_info 已扣除模型参数/优化器占用。
        估算公式：
          可用显存 = free_GB - 2.0 GB（headroom，预留给激活/梯度峰值）
          每 batch=8 约消耗 0.5 GB → batch = floor(可用 / 0.5) × 8
          结果取不超过上限的最大 2 的幂（1/2/4/8/16/32/64）。
        """
        if not torch.cuda.is_available():
            return 1
        try:
            free_bytes, _ = torch.cuda.mem_get_info(self.device)
        except Exception:
            return self.LM_BATCH_SIZE   # 查询失败则回退到类默认
        free_gb  = free_bytes / 1024 ** 3
        headroom = 2.0
        avail_gb = max(0.0, free_gb - headroom)
        # 每 batch=8 约需 0.5 GB（fp16 单层 attn map × 4 保守系数）
        raw_batch = int(avail_gb / 0.5) * 8
        raw_batch = max(raw_batch, 1)
        # 取最近的（不超过 raw_batch 的）2 的幂，上限 64
        batch = 1
        for p in [2, 4, 8, 16, 32, 64]:
            if p <= raw_batch:
                batch = p
        return batch

    def _init_lm_batch_size(self):
        """三层策略确定实际 LM 批次大小，并打印说明。

        优先级：
          1. cfg.lm.train.p4_lm_batch_size > 0  → 直接使用（强制固定）
          2. = 0（auto）                         → 按 GPU 显存估算
          3. 估算结果再用 OOM 自动减半兜底（见 _run_lm_batches）
        """
        cfg_val = self._cfg_lm_batch_size
        if cfg_val > 0:
            self._lm_batch_size = cfg_val
            src = f"cfg 固定值"
        else:
            self._lm_batch_size = self._estimate_lm_batch_size()
            try:
                free_gb = torch.cuda.mem_get_info(self.device)[0] / 1024 ** 3
            except Exception:
                free_gb = -1.0
            src = f"自动估算（GPU 可用 {free_gb:.1f} GB）"

        print(f"[P4] LM_BATCH_SIZE={self._lm_batch_size}  "
              f"来源={src}  "
              f"fp16={'启用' if self.use_amp else '禁用'}  "
              f"p4_use_gpt={self.p4_use_gpt}  "
              f"refresh_interval={self._lm_refresh_interval}")

    def _run_lm_batches(
        self,
        node_ids_cpu: list,
        with_grad: bool = False,
    ) -> list:
        """统一的 LM 批次循环，支持 OOM 自动减半并重试。

        参数
        ----
        node_ids_cpu : list[int]  需要推理的节点全局索引（必须是 CPU 可索引序列）
        with_grad    : bool
            True  → 保留梯度（训练节点）；cls tensor 留在 GPU，float32
            False → 无梯度（缓存刷新）；cls tensor 移至 CPU，float32

        返回
        ----
        list of Tensor，每个元素对应一个 mini-batch 的 cls embedding。
        调用方用 torch.cat() 合并。

        OOM 策略
        --------
        捕获 torch.cuda.OutOfMemoryError，将 self._lm_batch_size 减半，
        清空缓存后从头重试，直到 batch_size=1 仍 OOM 才向上抛出。
        """
        while True:
            try:
                embs = []
                n = len(node_ids_cpu)
                bs = self._lm_batch_size
                for start in range(0, n, bs):
                    idx = node_ids_cpu[start: start + bs]
                    batch = {k: v[idx].to(self.device)
                             for k, v in self.token_ids.items()}
                    if with_grad:
                        with torch.amp.autocast('cuda', enabled=self.use_amp):
                            out = self.lm_model.bert_encoder(
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                return_dict=True, output_hidden_states=True,
                            )
                        last_hidden = out["hidden_states"][-1].float()
                        cls = self.lm_model.dropout(
                            last_hidden).permute(1, 0, 2)[0]
                        if self.lm_model.feat_shrink:
                            cls = self.lm_model.feat_shrink_layer(cls)
                        embs.append(cls)                  # GPU, grad
                    else:
                        with torch.no_grad(), \
                             torch.amp.autocast('cuda', enabled=self.use_amp):
                            out = self.lm_model.bert_encoder(
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                return_dict=True, output_hidden_states=True,
                            )
                        cls = self.lm_model.dropout(
                            out["hidden_states"][-1]
                        ).permute(1, 0, 2)[0]
                        if self.lm_model.feat_shrink:
                            cls = self.lm_model.feat_shrink_layer(cls)
                        embs.append(cls.cpu().float())    # CPU, no grad
                return embs

            except torch.cuda.OutOfMemoryError:
                if self._lm_batch_size <= 1:
                    raise   # 已到最小，无法继续减半
                old = self._lm_batch_size
                self._lm_batch_size = max(1, self._lm_batch_size // 2)
                torch.cuda.empty_cache()
                print(f"[P4] ⚠ OOM！LM_BATCH_SIZE 自动减半: {old} → {self._lm_batch_size}，重试…")

    # ──────────────────────────────────────────────────────────────────────────
    # LM 推理（无梯度，用于缓存刷新）
    # ──────────────────────────────────────────────────────────────────────────

    def _lm_infer_all(self) -> torch.Tensor:
        """对全图所有节点做 LM 推理，返回 CPU float32 tensor [N, D]。

        使用 _run_lm_batches(with_grad=False) 进行推理，支持 OOM 自动减半。
        """
        self.lm_model.eval()
        node_ids = list(range(self.num_nodes))
        embs = self._run_lm_batches(node_ids, with_grad=False)
        return torch.cat(embs, dim=0)   # [N, D], CPU

    # ──────────────────────────────────────────────────────────────────────────
    # LM 前向（有梯度，仅用于训练节点）
    # ──────────────────────────────────────────────────────────────────────────

    def _lm_forward_train(self, node_indices: torch.Tensor) -> torch.Tensor:
        """对 node_indices 指定的节点用 LM 计算 embedding，梯度保留。

        返回 [len(node_indices), D]（GPU，float32）。
        使用 _run_lm_batches(with_grad=True)，OOM 时自动减半重试。

        node_indices 可以是 GPU tensor；内部自动转 CPU list 进行索引。
        """
        self.lm_model.train()
        node_ids = node_indices.cpu().tolist()
        embs = self._run_lm_batches(node_ids, with_grad=True)
        return torch.cat(embs, dim=0)   # [n_train, D], grad enabled, float32

    # ──────────────────────────────────────────────────────────────────────────
    # 特征矩阵构建（训练节点带梯度 + 其他节点用缓存）
    # ──────────────────────────────────────────────────────────────────────────

    def _build_features(self) -> torch.Tensor:
        """
        返回 [N, D] 特征矩阵：
          · train_mask 节点：由 LM 动态计算（梯度流经 LM → optimizer step）
          · 其余节点：使用 emb_cache（无梯度）
        利用 index_put（可微）将两部分合并，保证反向传播正确。
        """
        train_idx  = self.data.train_mask.nonzero(as_tuple=True)[0]   # [n_train]
        train_embs = self._lm_forward_train(train_idx)                 # [n_train, D]

        # index_put 是可微操作：
        #   features[train_idx] = train_embs  （梯度流经 train_embs）
        #   features[others]    = emb_cache   （无梯度）
        features = self.emb_cache.detach().clone()
        features = features.index_put((train_idx,), train_embs)
        return features   # [N, D]

    # ──────────────────────────────────────────────────────────────────────────
    # 训练 / 评估
    # ──────────────────────────────────────────────────────────────────────────

    def _train(self):
        self.model.train()
        self.gnn_optimizer.zero_grad()
        self.lm_optimizer.zero_grad()

        features = self._build_features()                         # [N, D], float32
        logits   = self.model(features, self.data.edge_index)
        loss     = self.loss_func(
            logits[self.data.train_mask],
            self.data.y[self.data.train_mask],
        )
        train_acc = self.evaluator(
            logits[self.data.train_mask],
            self.data.y[self.data.train_mask],
        )

        # GradScaler 统一处理 LM（fp16）和 GNN（fp32）的梯度
        # scaler.scale(loss).backward() 对 fp32 loss 等价于 loss.backward()
        # 但可保证 fp16 激活值的梯度正确缩放
        self.scaler.scale(loss).backward()
        self.scaler.step(self.gnn_optimizer)
        self.scaler.step(self.lm_optimizer)
        self.scaler.update()

        return loss.item(), train_acc

    @torch.no_grad()
    def _evaluate(self):
        """
        用当前缓存（可能略旧）做快速评估，供 early stopping 使用。
        """
        self.model.eval()
        logits  = self.model(self.emb_cache, self.data.edge_index)
        val_acc  = self.evaluator(
            logits[self.data.val_mask], self.data.y[self.data.val_mask])
        test_acc = self.evaluator(
            logits[self.data.test_mask], self.data.y[self.data.test_mask])
        return val_acc, test_acc, logits

    # ──────────────────────────────────────────────────────────────────────────
    # 主训练循环
    # ──────────────────────────────────────────────────────────────────────────

    @time_logger
    def train(self):
        for epoch in range(self.epochs):
            t0, es_str = time(), ""

            # 每隔 _lm_refresh_interval 轮刷新缓存（含第 0 轮已在 __init__ 初始化）
            if epoch > 0 and epoch % self._lm_refresh_interval == 0:
                self.emb_cache = self._lm_infer_all().to(self.device)

            loss, train_acc = self._train()
            val_acc, test_acc, _ = self._evaluate()

            if self.stopper is not None:
                es_flag, es_str = self.stopper.step(val_acc, self.model, epoch)
                if es_flag:
                    print(
                        f"[P4] Early stopped，从 epoch-{self.stopper.best_epoch} 加载模型"
                    )
                    break

            if epoch % LOG_FREQ == 0:
                print(
                    f"Epoch: {epoch}, Time: {time()-t0:.4f}, "
                    f"Loss: {loss:.4f}, TrainAcc: {train_acc:.4f}, "
                    f"ValAcc: {val_acc:.4f}, ES: {es_str}"
                )

        # 加载最优 GNN checkpoint
        if self.stopper is not None:
            self.model.load_state_dict(
                torch.load(self.stopper.path, weights_only=False)
            )

        return self.model

    # ──────────────────────────────────────────────────────────────────────────
    # 最终评估 & 保存（只保存 GNN，不保存 LM）
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def eval_and_save(self):
        """
        最终用最新 LM 权重对全图重推理，报告精度。
        仅保存 GNN checkpoint；LM 的微调结果不持久化。
        """
        torch.save(self.model.state_dict(), self.ckpt)

        # 最终全图 LM 推理（最新权重）
        final_cache = self._lm_infer_all().to(self.device)
        self.model.eval()
        logits   = self.model(final_cache, self.data.edge_index)
        val_acc  = self.evaluator(
            logits[self.data.val_mask], self.data.y[self.data.val_mask])
        test_acc = self.evaluator(
            logits[self.data.test_mask], self.data.y[self.data.test_mask])

        print(
            f"[{self.gnn_model_name} + LM-joint(P4)] "
            f"ValAcc: {val_acc:.4f}, TestAcc: {test_acc:.4f}\n"
        )
        return None, {"val_acc": val_acc, "test_acc": test_acc}
