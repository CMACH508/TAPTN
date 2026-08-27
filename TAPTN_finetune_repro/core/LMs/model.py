import torch
import torch.nn as nn
import numpy as np
from transformers import PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
from core.utils import init_random_state


# ─────────────────────────────────────────────────────────────────────────────
# 轻量级内容感知注意力池化
# ─────────────────────────────────────────────────────────────────────────────
class ChunkAttentionPool(nn.Module):
    """
    对滑窗产生的 n_chunks 个 CLS embedding 做内容感知（动态）加权平均。

    架构（加性注意力 / Bahdanau-style）
    ─────────────────────────────────────
      score_i = v · Tanh( W · cls_i )
        W : (hidden_dim → attn_dim)  ← 捕获语义特征
        v : (attn_dim → 1)           ← 学习"重要性方向"
      α_i = Softmax( [score_i] )     ← 对有效 chunk 归一化
      out = Σ_i α_i · cls_i

    参数量
    ──────
      W : hidden_dim × attn_dim  （默认 768×64 = 49 152）
      v : attn_dim               （默认 64）
      合计 ≈ 49 K  vs  6层TransformerEncoder ≈ 14 M

    动态性
    ──────
    权重完全由 chunk 内容决定：包含高置信度分类结论的 chunk（如 iter2 推理）
    其 CLS embedding 会触发更高的注意力分数，而主要含填充或低信息量内容
    的 chunk 得分自动降低，无需手动指定位置权重。

    可选：位置偏置（use_pos_bias=True）
      在注意力分数上叠加可学习的标量位置偏置 b_i：
        score_i += b_i
      参数量 +max_chunks（默认 32）。启用后兼顾内容 + 位置双重信号。
    """

    def __init__(self, hidden_dim: int, attn_dim: int = 64,
                 use_pos_bias: bool = False, max_chunks: int = 32):
        super().__init__()
        # W：将 CLS embedding 投影到低维注意力空间
        self.W = nn.Linear(hidden_dim, attn_dim, bias=True)
        # v：计算标量注意力分数
        self.v = nn.Linear(attn_dim, 1, bias=False)
        # 可选：可学习的位置偏置（初始化为 0，不干扰训练初期的内容注意力）
        self.use_pos_bias = use_pos_bias
        if use_pos_bias:
            self.pos_bias = nn.Parameter(torch.zeros(max_chunks))
        self._max_chunks = max_chunks

    def forward(self,
                chunk_embs: torch.Tensor,
                valid_masks: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        chunk_embs  : (B, n_chunks, H)  各 chunk 的 CLS embedding
        valid_masks : (B, n_chunks)     float，1=有效 token，0=全填充

        返回
        ----
        (B, H) 加权聚合后的句子 embedding
        """
        # (B, n_chunks, attn_dim) → (B, n_chunks, 1) → (B, n_chunks)
        scores = self.v(torch.tanh(self.W(chunk_embs))).squeeze(-1)

        # 叠加位置偏置（可选）
        if self.use_pos_bias:
            n = chunk_embs.size(1)
            scores = scores + self.pos_bias[:n].unsqueeze(0)

        # 无效 chunk（全填充）分数设为 -inf，不参与 softmax
        scores = scores.masked_fill(valid_masks == 0, float('-inf'))

        # 归一化注意力权重；若某样本所有 chunk 均为填充（极端情况），
        # softmax 输出 NaN → 用 nan_to_num 安全处理为 0
        weights = torch.softmax(scores, dim=-1)            # (B, n_chunks)
        weights = torch.nan_to_num(weights, nan=0.0)

        # 加权求和
        return (chunk_embs * weights.unsqueeze(-1)).sum(dim=1)  # (B, H)


# ─────────────────────────────────────────────────────────────────────────────

class BertClassifier(PreTrainedModel):
    def __init__(self, model, n_labels, dropout=0.0, seed=0, cla_bias=True, feat_shrink='',
                 use_gpt=False, chunk_pool='mean', chunk_attn_dim=64,
                 chunk_attn_pos_bias=False, chunk_stride=256,
                 use_llm_label=False, llm_label_mode='concat',
                 llm_logit_scale=3.0, dual_loss_weight=0.5,
                 use_gpt_soft_label=False, gpt_soft_label_kl_weight=0.3):
        """
        参数
        ----
        chunk_pool : str
            滑窗 chunk 的 CLS embedding 聚合方式，仅在 use_gpt=True 时有效。
            · 'mean'      — 等权均值（原始行为，向后兼容）
            · 'last'      — 只取最后一个有效 chunk（iter2 结论在末尾时高效）
            · 'tail_w'    — 位置线性加权（后面 chunk 权重线性递增）
            · 'softmax_w' — Softmax 位置加权（指数递增，更强调末尾）
            · 'attn'      — 内容感知动态注意力（推荐）
                            权重由每个 chunk 的 CLS embedding 动态计算，
                            无需手工指定位置偏好；参数量 ≈ 49K。
        chunk_attn_dim : int
            'attn' 模式下注意力隐层维度（默认 64）。
        chunk_attn_pos_bias : bool
            'attn' 模式下是否叠加可学习位置偏置（默认 False）。
            启用后在纯内容注意力基础上额外引入位置偏好先验。
        chunk_stride : int
            滑窗步长（默认 256，重叠 50%）。
            · 256 — 标准滑窗，相邻 chunk 重叠 256 token（默认）
            · 512 — 无重叠模式，与 split_seg 格式配合使用；
                    每段独立 tokenize 到 512 token 后拼接，
                    stride=512 确保各段各占一个完整 chunk，完全不互相污染。
        use_llm_label : bool
            是否在分类中融入 LLM 预测标签。
            True  → 以 llm_label_mode 指定的方式融合；
                    forward 时需接收 'llm_label_idx' 键（LongTensor, -1=未知）。
            False → 原始行为，分类头只接 CLS embedding（向后兼容）。
        llm_label_mode : str
            LLM one-hot 标签的融合方式（仅 use_llm_label=True 时有效）：
            · 'concat' — 旧行为：Linear([CLS ‖ one_hot], n_labels)
                         问题：W_cls(36K参数)易过拟合，可覆盖 one-hot 信号
            · 'add'    — logit 加法（推荐）：
                         logits = Linear(CLS, n_labels) + llm_logit_scale × one_hot
                         CLS 路径无 shortcut 被迫独立学习；one-hot 直接叠加到 logit；
                         仅新增 1 个可学习标量 llm_logit_scale；
                         exp(3.0)≈20x 概率提升，LM 若不够自信则 LLM 胜出
            · 'mix'    — 概率混合：
                         prob = (1-α)·softmax(Linear(CLS)) + α·one_hot
                         α = sigmoid(可学习参数)，初始 0.5；两种预测的凸组合
        llm_logit_scale : float
            'add' 模式下 one-hot 的初始缩放因子（可学习标量）。
            · 3.0 → 对应 ≈20x 的概率提升（推荐起点）
            · 0.0 → 等价于禁用 one-hot（退化为普通分类）
        dual_loss_weight : float
            'concat' 模式专用：双路径辅助 loss 的权重。
            总 loss = CE(CLS+one-hot) + dual_loss_weight × CE(CLS only)
            0.0 → 禁用；'add'/'mix' 模式下此参数被忽略
            （因为这两种模式的 CLS 路径本身就是独立路径）
        """
        # ★ 修复：在任何可学习模块（nn.Linear 等）创建之前先固定随机种子。
        # 原始代码在 self.classifier 创建之后才调用 init_random_state(seed)，
        # 导致分类头的权重被 get_cora_casestudy(cfg.seed) 中的 torch.manual_seed(cfg.seed)
        # 所污染，不同 cfg.seed 对应不同的分类头初始值，seed=6 时陷入退化局部最优。
        init_random_state(seed)
        super().__init__(model.config)
        self.bert_encoder = model
        self.dropout = nn.Dropout(dropout)
        self.feat_shrink = feat_shrink
        hidden_dim = model.config.hidden_size
        self.loss_func = nn.CrossEntropyLoss(
            label_smoothing=0.3, reduction='mean')

        if feat_shrink:
            self.feat_shrink_layer = nn.Linear(
                model.config.hidden_size, int(feat_shrink), bias=cla_bias)
            hidden_dim = int(feat_shrink)

        self.use_llm_label  = use_llm_label
        self.llm_label_mode = llm_label_mode if use_llm_label else 'none'
        self.n_labels       = n_labels

        if use_llm_label:
            if llm_label_mode == 'concat':
                # 旧行为：拼接后过分类头
                self.classifier = nn.Linear(hidden_dim + n_labels, n_labels, bias=cla_bias)
                self.dual_loss_weight = dual_loss_weight
                # 辅助路径（双loss）：仅 CLS → 独立分类头
                self.classifier_no_onehot = nn.Linear(hidden_dim, n_labels, bias=cla_bias)
            elif llm_label_mode in ('add', 'mix'):
                # CLS 路径干净，one-hot 在 logit/prob 层直接叠加
                self.classifier = nn.Linear(hidden_dim, n_labels, bias=cla_bias)
                self.dual_loss_weight = 0.0   # add/mix 本身 CLS 路径已独立
                if llm_label_mode == 'add':
                    # 可学习标量：one-hot logit 缩放因子
                    self.llm_logit_scale = nn.Parameter(
                        torch.tensor(float(llm_logit_scale)))
                else:  # 'mix'
                    # 可学习混合权重：sigmoid(0)=0.5 初始混合比
                    self.llm_mix_alpha_raw = nn.Parameter(torch.tensor(0.0))
            else:
                raise ValueError(f'未知 llm_label_mode: {llm_label_mode!r}')
        else:
            self.classifier = nn.Linear(hidden_dim, n_labels, bias=cla_bias)
            self.dual_loss_weight = 0.0

        # ── 方案C：GPT soft-label KL 损失 ─────────────────────────────────────
        # use_gpt_soft_label=True 时，forward 额外接收 gpt_soft_label (FloatTensor B×C)，
        # 计算 KL(model_prob || gpt_soft_label) 并加权加入总损失：
        #   loss = CE(logits, hard_labels) + kl_weight × KL(model || gpt_soft)
        # 全零行（提取失败的节点）自动跳过，不贡献 KL 损失。
        self.use_gpt_soft_label      = use_gpt_soft_label
        self.gpt_soft_label_kl_weight = gpt_soft_label_kl_weight

        # init_random_state(seed) 已移至 __init__ 最前面（见上方），此处删除重复调用。
        self.chunk_size = 512
        self.stride = chunk_stride      # 256（标准重叠）或 512（split_seg 无重叠）
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.aggregator = nn.TransformerEncoder(encoder_layer, num_layers=6, norm=nn.LayerNorm(hidden_dim))
        self.use_gpt = use_gpt
        self.chunk_pool = chunk_pool

        # 动态注意力池化模块（仅 chunk_pool='attn' 时使用）
        if chunk_pool == 'attn':
            self.chunk_attn_pool = ChunkAttentionPool(
                hidden_dim,
                attn_dim=chunk_attn_dim,
                use_pos_bias=chunk_attn_pos_bias,
            )
        #self._init_aggregator()

    def _init_aggregator(self):
        # Initialize weights for aggregator layers
        for module in self.aggregator.modules():
            if isinstance(module, nn.TransformerEncoderLayer):
                # Xavier initialization for the linear layers
                nn.init.xavier_uniform_(module.linear1.weight)
                nn.init.xavier_uniform_(module.linear2.weight)
                if module.linear1.bias is not None:
                    nn.init.zeros_(module.linear1.bias)
                if module.linear2.bias is not None:
                    nn.init.zeros_(module.linear2.bias)
                # Standard initialization for the attention layer
                nn.init.xavier_uniform_(module.self_attn.in_proj_weight)
                nn.init.xavier_uniform_(module.self_attn.out_proj.weight)
                if module.self_attn.in_proj_bias is not None:
                    nn.init.zeros_(module.self_attn.in_proj_bias)
                if module.self_attn.out_proj.bias is not None:
                    nn.init.zeros_(module.self_attn.out_proj.bias)

    def _needs_sliding_window(self, input_ids):
        """是否走滑窗+聚合路径（与 P2 相同机制）。

        · use_gpt=True：始终滑窗（原行为）
        · use_gpt=False + DeBERTa：整段单 forward（原 P1 行为，即使 token>512）
        · use_gpt=False + 其他 encoder：序列超过 max_position_embeddings 时
          自动滑窗，tokenization 仍与 DeBERTa 一致（不在 lm_trainer 额外截断）
        """
        if self.use_gpt:
            return True
        model_type = getattr(getattr(self.bert_encoder, 'config', None), 'model_type', '')
        if model_type == 'deberta':
            return False
        enc_max = getattr(self.bert_encoder.config, 'max_position_embeddings', 512)
        return input_ids.size(1) > enc_max

    def _pool_chunk_embeddings(self, chunk_embeddings, valid_masks, num_chunks):
        """滑窗 chunk 聚合（mean/last/tail_w/softmax_w/attn）。"""
        if self.chunk_pool == 'attn':
            return self.chunk_attn_pool(chunk_embeddings, valid_masks)
        if self.chunk_pool == 'last':
            last_valid = (valid_masks * torch.arange(
                num_chunks, device=valid_masks.device).float()).argmax(dim=1)
            return chunk_embeddings[
                torch.arange(chunk_embeddings.size(0), device=chunk_embeddings.device),
                last_valid]
        if self.chunk_pool in ('tail_w', 'softmax_w'):
            pos = torch.arange(num_chunks, device=chunk_embeddings.device).float()
            if self.chunk_pool == 'tail_w':
                pos_w = pos + 1.0
            else:
                pos_w = torch.exp(pos / max(num_chunks / 4.0, 1.0))
            pos_w = pos_w.unsqueeze(0) * valid_masks
            pos_w_sum = pos_w.sum(dim=1, keepdim=True).clamp(min=1e-9)
            pos_w = pos_w / pos_w_sum
            return (chunk_embeddings * pos_w.unsqueeze(-1)).sum(dim=1)
        return (chunk_embeddings * valid_masks.unsqueeze(-1)).sum(dim=1) \
            / valid_masks.sum(dim=1, keepdim=True)

    def _encode_sliding_window(self, input_ids, attention_mask, return_dict, apply_dropout=True):
        num_chunks = (input_ids.size(1) - self.chunk_size) // self.stride + 1
        chunk_embeddings = []
        chunk_masks = []
        for i in range(num_chunks):
            chunk_start = i * self.stride
            chunk_end = chunk_start + self.chunk_size
            chunk_input_ids = input_ids[:, chunk_start:chunk_end]
            chunk_attention_mask = attention_mask[:, chunk_start:chunk_end]
            outputs = self.bert_encoder(
                input_ids=chunk_input_ids,
                attention_mask=chunk_attention_mask,
                return_dict=return_dict,
                output_hidden_states=True)
            emb = outputs['hidden_states'][-1]
            if apply_dropout:
                emb = self.dropout(emb)
            cls_token_emb = emb.permute(1, 0, 2)[0]
            if self.feat_shrink:
                cls_token_emb = self.feat_shrink_layer(cls_token_emb)
            chunk_embeddings.append(cls_token_emb)
            chunk_masks.append(chunk_attention_mask.sum(dim=1) != 0)
        chunk_embeddings = torch.stack(chunk_embeddings, dim=1)
        valid_masks = torch.stack(chunk_masks, dim=1).float()
        return self._pool_chunk_embeddings(chunk_embeddings, valid_masks, num_chunks)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                return_dict=None,
                preds=None,
                llm_label_idx=None,    # LongTensor (B,)，-1=未知节点
                gpt_soft_label=None,   # FloatTensor (B, n_classes)，方案C软独热；全零行跳过
                node_id=None):         # 兼容 Trainer 可能传入的 node_id（不使用）
        if self._needs_sliding_window(input_ids):
            cls_token_emb = self._encode_sliding_window(
                input_ids, attention_mask, return_dict)
        else:
            outputs = self.bert_encoder(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        return_dict=return_dict,
                                        output_hidden_states=True)
            emb = self.dropout(outputs['hidden_states'][-1])
            cls_token_emb = emb.permute(1, 0, 2)[0]
            if self.feat_shrink:
                cls_token_emb = self.feat_shrink_layer(cls_token_emb)

        # ── 标签融合：按 llm_label_mode 将 one-hot 信号融入分类 ──────────────
        labels = labels.view(-1)   # 安全展平，防 batch_size=1 时 squeeze 去掉批次维

        if self.use_llm_label and llm_label_idx is not None:
            import torch.nn.functional as F
            B = cls_token_emb.size(0)
            onehot = torch.zeros(B, self.n_labels,
                                 device=cls_token_emb.device,
                                 dtype=cls_token_emb.dtype)
            valid = (llm_label_idx >= 0)
            if valid.any():
                onehot[valid] = F.one_hot(
                    llm_label_idx[valid].clamp(min=0),
                    num_classes=self.n_labels
                ).to(cls_token_emb.dtype)

            if self.llm_label_mode == 'concat':
                # ── 旧行为：拼接 one-hot 后过分类头 ──────────────────────────
                cls_with_onehot = torch.cat([cls_token_emb, onehot], dim=-1)
                logits = self.classifier(cls_with_onehot)
                loss_main = self.loss_func(logits, labels)
                if self.dual_loss_weight > 0:
                    logits_aux = self.classifier_no_onehot(cls_token_emb)
                    loss = loss_main + self.dual_loss_weight * self.loss_func(logits_aux, labels)
                else:
                    loss = loss_main

            elif self.llm_label_mode == 'add':
                # ── logit 加法：Linear(CLS) + scale × one_hot ────────────────
                # CLS 路径完全独立，one-hot 直接叠加到 logit
                # scale 可学习（初始值 llm_logit_scale），会随训练自动调整
                logits = self.classifier(cls_token_emb) + self.llm_logit_scale * onehot
                loss = self.loss_func(logits, labels)

            else:  # 'mix'
                # ── 概率混合：(1-α)·softmax(CLS) + α·one_hot ─────────────────
                # α = sigmoid(raw_param)，初始 0.5，训练中自适应
                alpha = torch.sigmoid(self.llm_mix_alpha_raw)
                prob_cls = F.softmax(self.classifier(cls_token_emb), dim=-1)  # (B, C)
                prob = (1.0 - alpha) * prob_cls + alpha * onehot              # (B, C)
                # 用 NLLLoss(log_prob) 等价于 CE 但支持非 logit 输入
                log_prob = torch.log(prob.clamp(min=1e-9))
                loss = F.nll_loss(log_prob, labels)
                logits = log_prob  # 仍为"越高越好"的量，与 argmax 逻辑一致
        else:
            logits = self.classifier(cls_token_emb)
            loss = self.loss_func(logits, labels)

        # ── 方案C：GPT soft-label KL 正则项 ──────────────────────────────────
        # KL(model_prob || gpt_soft) — 拉近模型预测分布与 GPT 软分布
        # 全零行（节点 GPT 提取失败）自动跳过，不贡献梯度。
        if self.use_gpt_soft_label and gpt_soft_label is not None:
            import torch.nn.functional as _F
            # model_log_prob: (B, C)  gpt_soft: (B, C)
            model_log_prob = _F.log_softmax(logits, dim=-1)
            gpt_soft = gpt_soft_label.to(logits.device, dtype=logits.dtype)
            # 有效行：gpt_soft 不全零（提取成功的节点）
            valid_rows = gpt_soft.sum(dim=1) > 0   # (B,)
            if valid_rows.any():
                # KL(P || Q) = sum(P * (log P - log Q)) ，此处 Q=model_log_prob, P=gpt_soft
                kl = _F.kl_div(
                    model_log_prob[valid_rows],        # (M, C) log-prob
                    gpt_soft[valid_rows],              # (M, C) target prob
                    reduction='batchmean',
                )
                loss = loss + self.gpt_soft_label_kl_weight * kl

        return TokenClassifierOutput(loss=loss, logits=logits)


class BertClaInfModel(PreTrainedModel):
    def __init__(self, model, emb, pred, feat_shrink='', use_gpt=False):
        super().__init__(model.config)
        self.bert_classifier = model
        self.emb, self.pred = emb, pred
        self.feat_shrink = feat_shrink
        self.loss_func = nn.CrossEntropyLoss(
            label_smoothing=0.3, reduction='mean')
        self.chunk_size = 512
        # 继承 BertClassifier 的 stride（训练 / 推理保持一致）
        self.stride = getattr(model, 'stride', 256)
        self.use_gpt = use_gpt
        # 继承 BertClassifier 的 chunk_pool 设置
        self.chunk_pool = getattr(model, 'chunk_pool', 'mean')
        # 继承 use_llm_label / llm_label_mode / n_labels
        self.use_llm_label  = getattr(model, 'use_llm_label', False)
        self.llm_label_mode = getattr(model, 'llm_label_mode', 'none')
        self.n_labels       = getattr(model, 'n_labels', 47)
        # 继承方案C软标签设置（推理时不需要计算 KL，但签名需兼容以防 Trainer 传入该字段）
        self.use_gpt_soft_label = getattr(model, 'use_gpt_soft_label', False)
        # 'attn' 模式：直接复用训练好的 ChunkAttentionPool（共享权重，无额外参数）
        if self.chunk_pool == 'attn':
            if not hasattr(model, 'chunk_attn_pool'):
                raise RuntimeError(
                    '[BertClaInfModel] chunk_pool="attn" 但 BertClassifier '
                    '未包含 chunk_attn_pool，请检查模型初始化参数。'
                )
            self.chunk_attn_pool = model.chunk_attn_pool

    @torch.no_grad()
    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                return_dict=None,
                node_id=None,
                llm_label_idx=None,    # LongTensor (B,)，-1=未知节点
                gpt_soft_label=None,   # FloatTensor (B, C)；推理时不参与计算，仅兼容接收
                use_gpt=False):
        if self.bert_classifier._needs_sliding_window(input_ids):
            cls_token_emb = self.bert_classifier._encode_sliding_window(
                input_ids, attention_mask, return_dict, apply_dropout=False)
        else:
            bert_outputs = self.bert_classifier.bert_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=return_dict,
                output_hidden_states=True)
            emb = bert_outputs['hidden_states'][-1]
            cls_token_emb = emb.permute(1, 0, 2)[0]
            if self.feat_shrink:
                cls_token_emb = self.bert_classifier.feat_shrink_layer(
                    cls_token_emb)

        # ── 推理：保存纯 CLS emb（不含 one-hot），然后按 llm_label_mode 计算 logits ──
        # emb 存储的是纯 CLS 向量，用于 GNN 输入特征（不含 one-hot/scale 信息）
        batch_nodes = node_id.cpu().numpy()
        self.emb[batch_nodes] = cls_token_emb.cpu().numpy().astype(np.float16)

        if self.use_llm_label and llm_label_idx is not None:
            import torch.nn.functional as F
            B = cls_token_emb.size(0)
            onehot = torch.zeros(B, self.n_labels,
                                 device=cls_token_emb.device,
                                 dtype=cls_token_emb.dtype)
            valid = (llm_label_idx >= 0)
            if valid.any():
                onehot[valid] = F.one_hot(
                    llm_label_idx[valid].clamp(min=0),
                    num_classes=self.n_labels
                ).to(cls_token_emb.dtype)

            if self.llm_label_mode == 'concat':
                cls_for_logit = torch.cat([cls_token_emb, onehot], dim=-1)
                logits = self.bert_classifier.classifier(cls_for_logit)
            elif self.llm_label_mode == 'add':
                scale = self.bert_classifier.llm_logit_scale
                logits = self.bert_classifier.classifier(cls_token_emb) + scale * onehot
            else:  # 'mix'
                alpha = torch.sigmoid(self.bert_classifier.llm_mix_alpha_raw)
                prob_cls = F.softmax(self.bert_classifier.classifier(cls_token_emb), dim=-1)
                prob = (1.0 - alpha) * prob_cls + alpha * onehot
                logits = torch.log(prob.clamp(min=1e-9))
        else:
            logits = self.bert_classifier.classifier(cls_token_emb)

        # Save prediction and embeddings to disk (memmap)
        self.pred[batch_nodes] = logits.cpu().numpy().astype(np.float16)

        # 同 BertClassifier：用 view(-1) 安全展平，防止 batch_size=1 时被 squeeze 掉批次维
        labels = labels.view(-1)
        loss = self.loss_func(logits, labels)

        return TokenClassifierOutput(loss=loss, logits=logits)
