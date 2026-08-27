import torch
import numpy as np

from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, IntervalStrategy
from core.LMs.model import BertClassifier, BertClaInfModel
from core.data_utils.dataset import Dataset
from core.data_utils.load import load_data
from core.utils import init_path, time_logger, lm_pretrained_path, lm_model_root, lm_ckpt_stem, lm_output_stem
import os

# Set the max_split_size_mb to a value greater than 20
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

def compute_metrics(p):
    from sklearn.metrics import accuracy_score
    pred, labels = p
    # HuggingFace 的 find_labels() 会将 forward 签名中含 "label" 子串的所有参数
    # （如 llm_label_idx）都识别为标签列，导致 label_ids 变成 tuple，
    # 此处统一取第一个元素（真实 ground-truth 标签）。
    if isinstance(labels, tuple):
        labels = labels[0]
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy}


class LMTrainer():
    def __init__(self, cfg):
        self.dataset_name = cfg.dataset
        self.seed = cfg.seed

        self.model_name = cfg.lm.model.name
        self.feat_shrink = cfg.lm.model.feat_shrink

        self.weight_decay = cfg.lm.train.weight_decay
        self.dropout = cfg.lm.train.dropout
        self.att_dropout = cfg.lm.train.att_dropout
        self.cla_dropout = cfg.lm.train.cla_dropout
        self.batch_size = cfg.lm.train.batch_size
        self.epochs = cfg.lm.train.epochs
        self.warmup_epochs = cfg.lm.train.warmup_epochs
        self.eval_patience = cfg.lm.train.eval_patience
        self.grad_acc_steps = cfg.lm.train.grad_acc_steps
        self.lr = cfg.lm.train.lr
        # fp16：默认 False（全精度，数值稳定）；P4 的 fp16 由 joint_trainer 独立控制
        self.fp16 = bool(getattr(cfg.lm.train, 'fp16', False))

        self.use_gpt = cfg.lm.train.use_gpt
        self.sem     = getattr(cfg.lm.train, 'sem', True)   # 默认 True（向后兼容）
        if self.use_gpt:
            self.use_gpt_str = "3" if self.sem else "3nosem"
        else:
            self.use_gpt_str = ""
        self.output_dir = lm_output_stem(cfg, self.dataset_name, self.use_gpt_str, self.seed)
        self.ckpt_dir = lm_ckpt_stem(cfg, self.dataset_name, self.use_gpt_str, self.seed)
        self.lm_root = lm_model_root(cfg)

        # Preprocess data
        data, num_classes, text = load_data(
            dataset=self.dataset_name, use_text=True, use_gpt=self.use_gpt, seed=self.seed,
            sem=self.sem, cfg=cfg)
        self.data = data
        self.num_nodes = data.y.size(0)
        self.n_labels = num_classes

        tokenizer = AutoTokenizer.from_pretrained(
            lm_pretrained_path(self.model_name, self.lm_root))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

        # ── split_seg 格式检测 ────────────────────────────────────────────────
        # load_product.py 的 text_fmt='split_seg' 会在段落间插入 SEG_SEP。
        # lm_trainer 检测到后切换为「分段独立 tokenize」模式，
        # 单段溢出策略由 cfg.lm.train.split_seg_overflow 控制：
        #
        #   'slide'    (默认) — 溢出段展开为多个 sub-chunk（内容完整保留）
        #     ① 不截断地 tokenize 每段
        #     ② n_chunks = ceil(max_seg_len / 512)，target = n_chunks × 512
        #     ③ 所有 sample pad 到 target（attention_mask=0 的 pad chunk 在池化时被屏蔽）
        #     ④ 拼接后总序列长 = Σ_seg (n_chunks_seg × 512)，按需扩展
        #
        #   'truncate' — 每段 tail 截断到 512 token（固定序列长 = n_segs × 512）
        #     ① truncation=True, max_length=512
        #     ② 显存占用最省，适合 reason_trunc_chars 已预截短的场景
        import math
        from core.data_utils.load_product import SEG_SEP
        _CHUNK_SIZE    = 512
        _seg_overflow  = getattr(cfg.lm.train, 'split_seg_overflow', 'slide')
        # !! 关键修正 !!
        # 不能仅检查 text[0]：product 数据集中只有 400/13482 节点(池节点)有 SEG_SEP，
        # 第0号 LCC 节点几乎必然是非池节点 → SEG_SEP in text[0] 永远为 False，
        # 导致 split_seg 分支从未激活，SEG_SEP 字节被直接喂给 tokenizer 产生乱码 token。
        # 修正：扫描全量 text，只要有任意一条含 SEG_SEP 就视为 split_seg 格式。
        _is_split_seg  = (
            self.use_gpt
            and isinstance(text, list)
            and len(text) > 0
            and any(SEG_SEP in t for t in text)
        )

        if _is_split_seg:
            all_segs   = [t.split(SEG_SEP) for t in text]
            max_n_segs = max(len(s) for s in all_segs)
            N          = len(text)

            seg_ids_list, seg_mask_list = [], []
            seg_chunk_counts = []   # 每段实际占用的 chunk 数（调试用）

            if _seg_overflow == 'truncate':
                # ── 硬截断：每段固定 1 chunk，tail 丢弃 ─────────────────────
                for seg_idx in range(max_n_segs):
                    seg_texts = [
                        s[seg_idx] if seg_idx < len(s) else ''
                        for s in all_segs
                    ]
                    toks = tokenizer(
                        seg_texts,
                        padding='max_length',
                        truncation=True,
                        max_length=_CHUNK_SIZE,
                    )
                    seg_ids_list.append(np.array(toks['input_ids'],      dtype=np.int64))
                    seg_mask_list.append(np.array(toks['attention_mask'], dtype=np.int64))
                    seg_chunk_counts.append(1)

            else:  # 'slide'（默认）
                # ── 滑窗展开：溢出段自动切为多个 sub-chunk ───────────────────
                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                for seg_idx in range(max_n_segs):
                    seg_texts = [
                        s[seg_idx] if seg_idx < len(s) else ''
                        for s in all_segs
                    ]
                    # 不截断、不 padding：保留完整 token 序列
                    toks = tokenizer(
                        seg_texts,
                        padding=False,
                        truncation=False,
                        return_attention_mask=True,
                    )
                    # 该段 batch 内最长 → 决定占几个 chunk
                    max_seg_len  = max(len(ids) for ids in toks['input_ids'])
                    n_chunks_seg = max(1, math.ceil(max_seg_len / _CHUNK_SIZE))
                    target_len   = n_chunks_seg * _CHUNK_SIZE
                    seg_chunk_counts.append(n_chunks_seg)

                    # 手动 pad 到 target_len
                    padded_ids   = np.full((N, target_len), pad_id,  dtype=np.int64)
                    padded_masks = np.zeros((N, target_len),          dtype=np.int64)
                    for j, (ids, mask) in enumerate(
                        zip(toks['input_ids'], toks['attention_mask'])
                    ):
                        l = min(len(ids), target_len)
                        padded_ids[j,   :l] = ids[:l]
                        padded_masks[j, :l] = mask[:l]

                    seg_ids_list.append(padded_ids)    # (N, n_chunks_seg × 512)
                    seg_mask_list.append(padded_masks)

            total_chunks = sum(seg_chunk_counts)
            print(f'[LMTrainer] split_seg 模式 (overflow={_seg_overflow}): '
                  f'{max_n_segs} 段/节点，各段 chunk 数={seg_chunk_counts}，'
                  f'总 chunk 数={total_chunks}，'
                  f'序列长={total_chunks * _CHUNK_SIZE} token，stride=512（无重叠）')

            # 沿 seq 维度拼接: [N, total_chunks * 512]
            X = {
                'input_ids':      np.concatenate(seg_ids_list,  axis=1),
                'attention_mask': np.concatenate(seg_mask_list, axis=1),
            }
            _chunk_stride = _CHUNK_SIZE   # 无重叠，每 512 token = 一个 chunk

        elif self.use_gpt:
            # 标准滑窗格式：完整序列不截断，stride=256（50% 重叠）
            X = tokenizer(text, padding=True, truncation=False)
            _chunk_stride = 256

        else:
            # use_gpt=False：直接过 LM，截断到 2048 防 OOM（与 DeBERTa 默认一致）
            # 非 DeBERTa 且序列超出 max_position_embeddings 时，由 BertClassifier
            # 内部自动切换为与 P2 相同的滑窗+聚合，不在此处引入额外截断规则。
            X = tokenizer(text, padding=True, truncation=True, max_length=2048)
            _chunk_stride = 256   # 供 BertClassifier 在需要滑窗时使用

        # ── LLM 预测标签（独热辅助输入） ──────────────────────────────────────
        # data.llm_label_idx 由 build_product_gpt_text 写入（use_gpt=True + product）；
        # 其他数据集或 use_gpt=False 时不存在，此时 _llm_label_idx=None（禁用此特性）。
        _llm_label_idx = getattr(data, 'llm_label_idx', None)
        _use_llm_label = (
            getattr(cfg.lm.train, 'use_llm_label', False)
            and _llm_label_idx is not None
        )
        if _use_llm_label:
            _llm_label_mode   = getattr(cfg.lm.train, 'llm_label_mode',          'add')
            _llm_logit_scale  = getattr(cfg.lm.train, 'llm_logit_scale',          3.0)
            _llm_feat_shrink  = getattr(cfg.lm.train, 'llm_label_feat_shrink',    0)
            _llm_weight_decay = getattr(cfg.lm.train, 'llm_label_weight_decay',   0.01)
            _llm_dual_loss_w  = getattr(cfg.lm.train, 'llm_label_dual_loss_weight', 0.5)
            if _llm_feat_shrink:
                self.feat_shrink = str(int(_llm_feat_shrink))
            if _llm_weight_decay:
                self.weight_decay = _llm_weight_decay
            self._dual_loss_weight = _llm_dual_loss_w if _llm_label_mode == 'concat' else 0.0

            _mode_desc = {
                'concat': f'concat([CLS, one_hot])→Linear  dual_loss_w={_llm_dual_loss_w}',
                'add':    f'Linear(CLS)+scale×one_hot  scale_init={_llm_logit_scale:.1f}（可学习）',
                'mix':    f'(1-α)·softmax(Linear(CLS))+α·one_hot  α_init=0.5（可学习）',
            }.get(_llm_label_mode, _llm_label_mode)
            print(f'[LMTrainer] use_llm_label=True [{_llm_label_mode}]: '
                  f'将 LLM 标签 one-hot（{self.n_labels}维）以 "{_llm_label_mode}" 方式融合')
            print(f'             模式说明: {_mode_desc}')
            print(f'[LMTrainer] use_llm_label 专用超参: '
                  f'feat_shrink={self.feat_shrink or "无"}, '
                  f'weight_decay={self.weight_decay:.4f}')
        else:
            _llm_label_idx    = None   # 统一为 None，Dataset/model 不启用
            _llm_label_mode   = 'none'
            _llm_logit_scale  = 3.0
            self._dual_loss_weight = 0.0

        # ── 方案C：GPT soft-label 辅助 KL 损失（仅 cora）────────────────────────
        # data.gpt_soft_label 由 load.py 在 use_gpt_soft_label=True 时写入；
        # 其他数据集或该参数为 False 时不存在，此处统一检测。
        _gpt_soft_label_tensor = getattr(data, 'gpt_soft_label', None)
        _use_gpt_soft_label = (
            getattr(cfg.lm.train, 'use_gpt_soft_label', False)
            and _gpt_soft_label_tensor is not None
        )
        _gpt_soft_label_weight = float(
            getattr(cfg.lm.train, 'gpt_soft_label_weight', 0.3)
        )
        if _use_gpt_soft_label:
            _hit = (_gpt_soft_label_tensor.sum(dim=1) > 0).sum().item()
            print(f'[LMTrainer] use_gpt_soft_label=True: '
                  f'KL 权重={_gpt_soft_label_weight:.2f}  '
                  f'有效节点: {_hit}/{len(_gpt_soft_label_tensor)}')
        else:
            _gpt_soft_label_tensor = None   # 统一 None，Dataset/model 不启用

        dataset = Dataset(X, data.y.tolist(),
                          llm_label_idx=_llm_label_idx,
                          gpt_soft_label=_gpt_soft_label_tensor)
        self.inf_dataset = dataset

        self.train_dataset = torch.utils.data.Subset(
            dataset, self.data.train_mask.nonzero().squeeze().tolist())
        self.val_dataset = torch.utils.data.Subset(
            dataset, self.data.val_mask.nonzero().squeeze().tolist())
        self.test_dataset = torch.utils.data.Subset(
            dataset, self.data.test_mask.nonzero().squeeze().tolist())

        # Define pretrained tokenizer and model
        bert_model = AutoModel.from_pretrained(
            lm_pretrained_path(self.model_name, self.lm_root))
        # chunk_pool：控制滑窗 CLS embedding 聚合策略（use_gpt=True 时生效）
        # · 'mean'      — 等权均值（原始行为，向后兼容）
        # · 'last'      — 只取最后一个有效 chunk
        # · 'tail_w'    — 位置线性加权（后 chunk 权重更大）
        # · 'softmax_w' — Softmax 位置加权
        # · 'attn'      — 内容感知动态注意力（推荐，~49K 额外参数）
        _chunk_pool       = getattr(cfg.lm.train, 'chunk_pool',           'tail_w')
        _chunk_attn_dim   = getattr(cfg.lm.train, 'chunk_attn_dim',       64)
        _chunk_attn_pbias = getattr(cfg.lm.train, 'chunk_attn_pos_bias',  False)
        self.model = BertClassifier(bert_model,
                                    n_labels=self.n_labels,
                                    feat_shrink=self.feat_shrink,
                                    use_gpt=self.use_gpt,
                                    chunk_pool=_chunk_pool,
                                    chunk_attn_dim=_chunk_attn_dim,
                                    chunk_attn_pos_bias=_chunk_attn_pbias,
                                    chunk_stride=_chunk_stride,
                                    use_llm_label=_use_llm_label,
                                    llm_label_mode=_llm_label_mode,
                                    llm_logit_scale=_llm_logit_scale,
                                    dual_loss_weight=self._dual_loss_weight,
                                    use_gpt_soft_label=_use_gpt_soft_label,
                                    gpt_soft_label_kl_weight=_gpt_soft_label_weight)

        # prev_ckpt = f'prt_lm/{self.dataset_name}/{self.model_name}.ckpt'
        # if self.use_gpt_str and os.path.exists(prev_ckpt):
        #     print("Initialize using previous ckpt...")
        #     self.model.load_state_dict(torch.load(prev_ckpt))

        self.model.config.dropout = self.dropout
        self.model.config.attention_dropout = self.att_dropout
        self.emb_dim = int(self.feat_shrink) if self.feat_shrink else bert_model.config.hidden_size

        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)
        print(f"\nNumber of parameters: {trainable_params}")

    @time_logger
    def train(self):
        # ── 检测实际可见 GPU 数，动态调整 per-device batch size ──────────────
        # 设计原则：self.batch_size 视为目标「全局 batch size」（单卡等价 per-device）。
        # 多卡 DataParallel 时，按比例缩减 per_device_batch_size，使
        #   per_device_bs × num_gpus ≈ self.batch_size（全局 batch 不变）。
        # 这样无论单卡首次还是多卡 OOM 重试，梯度期望相同，训练行为一致。
        _visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
        if _visible and _visible != '-1':
            _num_gpus = max(1, len([x for x in _visible.split(',') if x.strip().isdigit()]))
        else:
            _num_gpus = max(1, torch.cuda.device_count())

        per_device_bs = max(1, self.batch_size // _num_gpus)
        global_bs     = per_device_bs * _num_gpus   # 实际全局 batch（≤ self.batch_size）

        if _num_gpus > 1:
            print(f'[LMTrainer] 多卡 DataParallel: {_num_gpus} 张 GPU')
            print(f'[LMTrainer] batch_size 调整: per_device {self.batch_size} → {per_device_bs}'
                  f'  (全局 {global_bs}，目标 {self.batch_size})')
        else:
            per_device_bs = self.batch_size  # 单卡：不做改动

        print(f'[LMTrainer] 精度模式: {"fp16 混合精度" if self.fp16 else "fp32 全精度（默认）"}  '
              f'（命令行覆盖: lm.train.fp16 True）')

        # ── 步数计算：始终基于目标全局 batch（self.batch_size × 4），
        #    与实际 GPU 数无关，确保 eval_steps / warmup_steps 跨 GPU 数一致。 ──
        eq_batch_size = self.batch_size * 4
        train_steps   = self.num_nodes // eq_batch_size + 1
        eval_steps    = self.eval_patience // eq_batch_size
        warmup_steps  = int(self.warmup_epochs * train_steps)

        # Define Trainer
        args = TrainingArguments(
            output_dir=self.output_dir,
            do_train=True,
            do_eval=True,
            eval_steps=eval_steps,
            evaluation_strategy=IntervalStrategy.STEPS,
            save_steps=eval_steps,
            learning_rate=self.lr,
            weight_decay=self.weight_decay,
            save_total_limit=1,
            load_best_model_at_end=True,
            gradient_accumulation_steps=self.grad_acc_steps,
            per_device_train_batch_size=per_device_bs,
            per_device_eval_batch_size=per_device_bs * 8,
            warmup_steps=warmup_steps,
            num_train_epochs=self.epochs,
            dataloader_num_workers=1,
            fp16=self.fp16,
            dataloader_drop_last=True,
            # HuggingFace 的 find_labels() 将 forward 签名中含 "label" 子串的所有参数
            # 都当作标签列（如 llm_label_idx），导致 compute_metrics 收到 tuple labels。
            # 此处显式指定 label_names，确保只有 "labels" 被当作真实标签。
            label_names=["labels"],
        )
        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=compute_metrics,
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        # Train pre-trained model
        # !! 关键 !! 必须传 resume_from_checkpoint=False，防止 Trainer 自动从 output_dir
        # 中残留的旧 checkpoint 恢复（旧 checkpoint 可能来自不同 chunk_pool 配置，
        # 缺少 chunk_attn_pool 等权重，导致这些权重保持随机初始化状态，
        # 推理阶段精度崩塌至接近随机水平）。
        self.trainer.train(resume_from_checkpoint=False)
        torch.save(self.model.state_dict(), init_path(f"{self.ckpt_dir}.ckpt"))
        print(f'LM saved to {self.ckpt_dir}.ckpt')

    @time_logger
    @torch.no_grad()
    def eval_and_save(self):
        _emb_dim = self.emb_dim
        emb = np.memmap(init_path(f"{self.ckpt_dir}.emb"),
                        dtype=np.float16,
                        mode='w+',
                        shape=(self.num_nodes, _emb_dim))
        pred = np.memmap(init_path(f"{self.ckpt_dir}.pred"),
                         dtype=np.float16,
                         mode='w+',
                         shape=(self.num_nodes, self.n_labels))

        inf_model = BertClaInfModel(
            self.model, emb, pred, feat_shrink=self.feat_shrink, use_gpt=self.model.use_gpt)
        inf_model.eval()
        inference_args = TrainingArguments(
            output_dir=self.output_dir,
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=self.batch_size*8,
            dataloader_drop_last=False,
            dataloader_num_workers=1,
            fp16_full_eval=self.fp16,
            # !! 关键 !! 与训练阶段保持一致，防止 HuggingFace find_labels() 把
            # llm_label_idx（含 "label" 子串）也识别为标签列，导致 prediction_step
            # 将其从 inputs 中提取到 labels 元组、不再传给 model(**inputs)。
            # 一旦 llm_label_idx=None 落入 model 的 forward，LLM 混合完全失效，
            # 退化成纯 CLS 预测，测试精度崩塌。
            label_names=["labels"],
        )

        trainer = Trainer(model=inf_model, args=inference_args)
        trainer.predict(self.inf_dataset)
        if "ogbn" in self.dataset_name:
            from ogb.nodeproppred import Evaluator
            _evaluator = Evaluator(name=self.dataset_name)
        else:
            from core.GNNs.gnn_utils import Evaluator
            _evaluator = Evaluator(name=self.dataset_name)

        def evaluator(preds, labels): return _evaluator.eval({
            "y_true": torch.tensor(labels).view(-1, 1),
            "y_pred": torch.tensor(preds).view(-1, 1),
        })["acc"]

        def eval(x): return evaluator(
            np.argmax(pred[x], -1), self.data.y[x])

        train_acc = eval(self.data.train_mask)
        val_acc = eval(self.data.val_mask)
        test_acc = eval(self.data.test_mask)
        print(
            f'[LM] TrainAcc: {train_acc:.4f}, ValAcc: {val_acc:.4f}, TestAcc: {test_acc:.4f}\n')
        return {'TrainAcc': train_acc, 'ValAcc': val_acc, 'TestAcc': test_acc}
