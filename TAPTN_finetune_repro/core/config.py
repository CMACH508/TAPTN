import os
import argparse
from yacs.config import CfgNode as CN


def set_cfg(cfg):

    # ------------------------------------------------------------------------ #
    # Basic options
    # ------------------------------------------------------------------------ #
    # Dataset name
    cfg.dataset = 'wisconsin'
    # Cuda device number, used for machine with multiple gpus
    cfg.device = 2
    # Whether fix the running seed to remove randomness
    cfg.seed = None
    # Number of runs with random init
    if cfg.dataset in ['cora']:
        cfg.runs = 4
    elif cfg.dataset in ['arxiv_2023']:
        cfg.runs = 3
    else:
        cfg.runs = 1
    cfg.gnn = CN()
    cfg.lm = CN()

    # ------------------------------------------------------------------------ #
    # GNN Model options
    # ------------------------------------------------------------------------ #
    cfg.gnn.model = CN()
    # GNN model name
    cfg.gnn.model.name = 'GAT'
    # Number of gnn layers
    cfg.gnn.model.num_layers = 2
    # Hidden size of the model
    if cfg.gnn.model.name in ['SAGE','GAT']:
        cfg.gnn.model.hidden_dim = 1024
    cfg.gnn.model.hidden_dim = 128

    # ------------------------------------------------------------------------ #
    # GNN Training options
    # ------------------------------------------------------------------------ #
    cfg.gnn.train = CN()
    # The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights
    cfg.gnn.train.weight_decay = 0.0
    # Maximal number of epochs
    cfg.gnn.train.epochs = 200
    #cfg.gnn.train.epochs = 1000
    # Node feature type, options: ogb, TA, P, E
    cfg.gnn.train.feature_type = 'TA'
    # Number of epochs with no improvement after which training will be stopped
    cfg.gnn.train.early_stop = 50
    #cfg.gnn.train.early_stop = 200
    # Base learning rate
    cfg.gnn.train.lr = 0.01
    # L2 regularization, weight decay
    cfg.gnn.train.wd = 0.0
    # Dropout rate
    cfg.gnn.train.dropout = 0.0

    # ------------------------------------------------------------------------ #
    # LM Model options
    # ------------------------------------------------------------------------ #
    cfg.lm.model = CN()
    # LM model name
    cfg.lm.model.name = 'microsoft/deberta-base'
    cfg.lm.model.feat_shrink = ""
    # Local HuggingFace weight root (TAPTN_PRETRAINED, else TAPTN_ASSETS/pretrained)
    _assets = os.environ.get('TAPTN_ASSETS', '')
    _pretrained = os.environ.get('TAPTN_PRETRAINED')
    if _pretrained:
        cfg.lm.model.root = _pretrained
    elif _assets:
        cfg.lm.model.root = os.path.join(_assets, 'pretrained')
    else:
        cfg.lm.model.root = 'pretrained'
    # 跨模型实验产物隔离子路径（空 = 写入 prt_lm/、output/ 根下，与 DeBERTa 默认管线相同）
    # 非空示例：crosslm/FacebookAI_roberta-base_20260616_071332
    cfg.lm.model.artifact_root = ''

    # ------------------------------------------------------------------------ #
    # LM Training options
    # ------------------------------------------------------------------------ #
    cfg.lm.train = CN()
    #  Number of samples computed once per batch per device
    cfg.lm.train.batch_size = 1
    # Number of training steps for which the gradients should be accumulated
    cfg.lm.train.grad_acc_steps = 1
    # Base learning rate
    cfg.lm.train.lr = 2e-5
    # Maximal number of epochs
    cfg.lm.train.epochs = 15
    # The number of warmup steps
    cfg.lm.train.warmup_epochs = 0.6
    # Number of update steps between two evaluations
    cfg.lm.train.eval_patience = 5000
    # The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights
    cfg.lm.train.weight_decay = 0.0
    # The dropout probability for all fully connected layers in the embeddings, encoder, and pooler
    cfg.lm.train.dropout = 0.3
    # The dropout ratio for the attention probabilities
    cfg.lm.train.att_dropout = 0.1
    # The dropout ratio for the classifier
    cfg.lm.train.cla_dropout = 0.4
    # Whether or not to use the gpt responses (i.e., explanation and prediction) as text input
    # If not, use the original text attributes (i.e., title and abstract)
    cfg.lm.train.use_gpt = False
    # Whether to use semantic (graph-structure-guided) LLM reasoning (True) or
    # structure-free LLM reasoning (False).  Only effective when use_gpt=True.
    #   sem=True  → guided   pkls  → output dir suffix "3"
    #   sem=False → noguide  pkls  → output dir suffix "3nosem"
    cfg.lm.train.sem = True
    # 若 prt_lm/<dataset><suffix>/<model>-seed<N>.ckpt 已存在：
    #   False (默认) — 跳过训练阶段，仅重新执行 eval_and_save（生成 .emb / .pred）。
    #                  适合批量补跑、续跑时避免重复消耗 GPU 时间。
    #   True         — 忽略已有 checkpoint，从原始预训练权重（DeBERTa-base）重新微调
    #                  并覆盖写入 ckpt / emb / pred。适合对同一 seed 做超参消融实验。
    cfg.lm.train.force_retrain = False
    # 混合精度训练开关（P1/P2/P3 LM 微调阶段）：
    #   False（默认）— 全精度 fp32 训练，数值稳定，适合小数据集（如 Cora batch_size=1）
    #   True         — fp16 混合精度，节省显存约 30%，适合 batch_size ≥ 4 的大数据集
    # 注意：P4 联合训练的 fp16 由 joint_trainer.py 中 self.use_amp 独立控制（默认 True），
    #       此参数仅作用于 P1/P2/P3 的 HuggingFace Trainer 训练和推理阶段。
    # 命令行覆盖示例：python -m core.trainLM ... lm.train.fp16 True
    cfg.lm.train.fp16 = False
    # 滑窗 chunk CLS embedding 聚合策略（use_gpt=True 时有效）：
    #   'mean'      — 等权均值（原始行为）
    #   'last'      — 只取最后一个有效 chunk（iter2 结论在末尾时效果好）
    #   'tail_w'    — 位置线性加权（后面 chunk 权重线性递增；推荐）
    #   'softmax_w' — Softmax 位置加权（指数递增；末尾 chunk 权重更突出）
    #   'attn'      — 内容感知动态注意力（推荐；权重随 chunk 内容自适应调整）
    #                 架构：Linear(H→attn_dim) → Tanh → Linear(attn_dim→1) → Softmax
    #                 参数量 ≈ H×attn_dim ≈ 49K（vs 6层TransformerEncoder的14M）
    # 诊断结果：product sem 文本平均 4.5 个 chunk，iter2 结论在 chunk 2-3（96%样本）。
    # 等权均值时 iter2 信号仅占 22%；attn/tail_w 可显著提升 iter2 信号权重。
    cfg.lm.train.chunk_pool = 'attn'
    # 'attn' 模式下注意力隐层维度（默认 64；可设 128 增强表达但参数翻倍）
    cfg.lm.train.chunk_attn_dim = 128
    # 'attn' 模式下是否在注意力分数上叠加可学习位置偏置（默认 False）
    # True：兼顾内容 + 位置双重信号；适合 iter2 位置相对固定的场景
    cfg.lm.train.chunk_attn_pos_bias = False
    # split_seg 模式下，单段 token 数超过 chunk_size=512 时的溢出处理策略：
    #   'slide'    (默认) — 对溢出段以 stride=chunk_size 无重叠滑窗展开为多个
    #                       sub-chunk，内容完整保留；
    #                       总序列长 = Σ_seg ceil(max_seg_len/512) × 512。
    #                       适合推理文本较长且不希望丢失内容的场景。
    #   'truncate'        — 对溢出段 tail 截断到 512 token，超出部分丢弃；
    #                       总序列长固定 = n_segs × 512（固定批次大小，显存最省）。
    #                       适合显存紧张或 reason_trunc_chars 已将文本预截短的场景。
    cfg.lm.train.split_seg_overflow = 'slide'

    # ------------------------------------------------------------------------ #
    # Product dataset – LLM reasoning options
    # ------------------------------------------------------------------------ #
    cfg.product = CN()
    _PKL_HOME = os.path.join(_assets, 'pkls') if _assets else 'pkls'

    # sem=True  (guided, structure-aware)
    cfg.product.channel_pkls = [
        f'{_PKL_HOME}/product_hop1_noanon_guide_llama-3.3-70b-instruct_2.pkl',
        f'{_PKL_HOME}/product_hop2_noanon_noguide_llama-3.3-70b-instruct.pkl'
    ]
    cfg.product.iter2_pkl = (
        f'{_PKL_HOME}/product_hop1_noanon_guide_llama-3.3-70b-instruct_iter2_3.pkl'
    )
    cfg.product.channels_node_set = 'test'
    cfg.product.show_consensus = False

    # sem=False  (no-guide, structure-free; single source, no iter2)
    cfg.product.nosem_channel_pkls = [
        f'{_PKL_HOME}/product_hop2_noanon_noguide_llama-3.3-70b-instruct.pkl',
    ]
    cfg.product.nosem_iter2_pkl = ''   # 空字符串表示不使用 iter2

    # 每段 reason 的最大字符数（0 = 不截断）。
    # 诊断发现 product 的 LLM 推理文本平均 1526 tokens（4.5 个滑窗 chunk），
    # iter2 关键分类标签位于文本 56% 处，经均值聚合后信号仅占 22% 权重。
    # 建议值：
    #   1200 ≈ 300 tokens  → iter2标签进入chunk0的比例 79%，保留较多推理上下文（推荐）
    #    600 ≈ 150 tokens  → iter2标签进入chunk0的比例 91%，更高信噪比但截断更多
    #      0 = 不截断（向后兼容旧行为，iter2仅5%在chunk0）
    cfg.product.reason_trunc_chars = 0  # ≈300 tokens；与 load_product.REASON_TRUNC_CHARS 对齐

    # title 截断（字符数，0 = 不截断）
    # 实测 title+content 分布：avg=162t, p95=448t, p99=842t, max=48972t（0.2% 极端异常）
    # · result_first 格式下 title 本身是分类信号（"USB Hub"→电子），建议保留，设为 0
    # · split_seg 格式下若遇极端异常节点会导致 OOM（最大 96 chunks），建议设为 2000（≈500t）
    # · 2000 chars 覆盖 p99 节点（842t=3368chars），仅截断最极端的 0.02% 异常值
    cfg.product.title_trunc_chars = 2000   # 0 = 不截断（result_first 推荐）

    # ── text_fmt：仅控制文本排布结构 ─────────────────────────────────────
    #
    #   'result_first'  (默认/新格式)
    #     结论前置：分类标签紧跟 section 标题，推理文本在后。
    #
    #   'reason_first'  (旧格式/向后兼容)
    #     推理在前，分类标签在后（actor/product 原始行为）。
    #
    #   'split_seg'  (强制分离格式)
    #     result / reason / 不同 iter 各自独立 tokenize 到 512 token；
    #     stride=512（无重叠），每段各占一个完整 BERT chunk。
    #     示例（sem=True, include_iter1=True，最多 5 段）：
    #       [Seg0] Title and Abstract: ...               ← chunk 0
    #       [Seg1] Initial Classification: {result}      ← chunk 1（iter1 结论）
    #       [Seg2] Initial Reasoning: {reason[:trunc]}   ← chunk 2（iter1 推理）
    #       [Seg3] Refined Classification: {result}      ← chunk 3（iter2 结论）
    #       [Seg4] Refined Reasoning: {reason[:trunc]}   ← chunk 4（iter2 推理）
    #     示例（sem=False，最多 3 段）：
    #       [Seg0] Title and Abstract: ...               ← chunk 0
    #       [Seg1] Expert Classification: {result}       ← chunk 1
    #       [Seg2] Expert Reasoning: {reason[:trunc]}    ← chunk 2
    # 历史实验结论：
    #   result_first → 85%（最佳）：分类标签置前，chunk0 天然捕获，无需注意力学习 chunk 选择
    #   split_seg    → 75-81%：title 段可能长达 12 chunks（Amazon 商品描述无截断），
    #                  有效 iter2 信号永远在最后 1/13 chunk，240 样本下注意力无法收敛
    # 建议：保持 result_first；若实验 split_seg，务必同步截断 title（title_trunc_chars）
    cfg.product.text_fmt = 'split_seg'

    # ── include_iter1：控制是否展示 iter1/Initial Classification 段 ──────
    # · sem=True  + include_iter1=True  (默认): 展示 iter1 + iter2 两段
    # · sem=True  + include_iter1=False        : 跳过 iter1，仅展示 iter2
    # · sem=False (noguide)                    : 此参数无效，始终展示 Expert Classification
    cfg.product.include_iter1 = False

    # ── split_seg_merge_cls_reason：方案B分片格式（split_seg 专用） ──────────
    # False (默认) — 结论 / 推理各占独立 chunk（原始 split_seg 行为）：
    #   [Seg0] Title
    #   [Seg1] Refined Classification: {s2}          ← 仅结论（极短，易退化）
    #   [Seg2] Refined Reasoning: {r2[:trunc]}        ← 仅推理
    #
    # True  (方案B) — 结论 + 推理合并为同一段（避免短结论退化 chunk）：
    #   [Seg0] Title
    #   [Seg1] Refined Classification: {s2}
    #           Refined Reasoning: {r2[:trunc]}       ← 结论在前，推理紧随，同一 chunk
    #
    # 推荐：若结论文本极短（avg 4.8 tokens），建议使用 True 以确保结论有足够上下文。
    # 仅在 text_fmt='split_seg' 时生效，其他 fmt 下此参数被忽略。
    cfg.product.split_seg_merge_cls_reason = True

    # ── use_llm_label：利用 LLM 预测标签 one-hot 辅助分类 ────────────────
    # 须在 cfg.lm.train 节下设置（适用于所有数据集）
    # True  → build_product_gpt_text 会从 iter2/nosem result 字段提取类别索引，
    #         存于 data.llm_label_idx（LongTensor, shape=[num_nodes], -1=未知）；
    #         对于 pkl 未覆盖的节点（idx=-1），one-hot 为全零（不提供先验）。
    # False → 原始行为，分类头只接 CLS embedding（向后兼容）。
    cfg.lm.train.use_llm_label = True

    # ── llm_label_mode：one-hot 融合方式 ─────────────────────────────────────
    #
    # 'concat'（旧行为）：
    #   classifier = Linear(hidden_dim + n_labels, n_labels)
    #   input = [CLS ‖ one_hot]
    #   问题：36K 参数的 W_cls 易过拟合，可覆盖 one-hot 信号
    #
    # 'add'（推荐，logit 加法）：
    #   classifier = Linear(hidden_dim, n_labels)   ← CLS 路径独立，无 shortcut
    #   logits = classifier(CLS) + llm_logit_scale × one_hot
    #   效果：
    #   · LLM 预测类的 logit 直接获得 exp(scale) 倍概率提升（scale=3.0 ≈ 20x）
    #   · CLS 路径必须产生"足够自信"的负值才能覆盖 LLM 预测（迫使文本学习）
    #   · one-hot 的贡献不经过任何全连接层，不参与 W_cls 过拟合
    #   · 仅新增 1 个可学习标量（llm_logit_scale），参数量极少
    #   · scale 初始值 3.0：P(LLM_class) ≈ 20×P(other) → 强先验；训练后自适应调整
    #
    # 'mix'（概率混合）：
    #   prob = (1-α) × softmax(classifier(CLS)) + α × one_hot
    #   logits = log(prob + ε)
    #   α = sigmoid(llm_mix_alpha_param)，可学习（初始化使 α ≈ 0.5）
    #   效果：概率层面混合，LLM 和 LM 各占一定比例，更平滑
    cfg.lm.train.llm_label_mode = 'mix'   # 'concat' | 'add' | 'mix'

    # llm_logit_scale：'add' 模式的 one-hot 缩放因子（可学习标量，初始值）
    #   · 3.0 → LLM 类 logit 提升 3.0，对应 softmax ≈ 20x 概率增益
    #   · 0.0 → 等同于禁用 one-hot（退化为普通分类）
    #   · 设为负值无意义；训练后若收敛到大正值说明 LLM 信号强
    cfg.lm.train.llm_logit_scale = 3.0

    # ── use_llm_label=True 时的专用超参 ───────────────────────────────────────
    # llm_label_feat_shrink（默认 0）
    #   !! 注意 !! feat_shrink 压缩发生在 ChunkAttentionPool 之前，
    #   在 split_seg 多 chunk 场景下会严重损害注意力质量（实测 -6%）。
    #   仅 text_fmt='result_first' 单序列场景可酌情启用小值（如 256）。
    #   对于 split_seg 或多 chunk 场景，必须设为 0。
    cfg.lm.train.llm_label_feat_shrink      = 0     # 0 = 不压缩
    cfg.lm.train.llm_label_weight_decay     = 0.01  # 覆盖全局 weight_decay
    cfg.lm.train.llm_label_dual_loss_weight = 0.5   # 辅助路径 loss 权重（'concat' 模式专用）

    # ── P4 联合训练文本选择 ───────────────────────────────────────────────────
    # p4_use_gpt（默认 False）：P4 联合训练时是否使用 GPT 增强文本
    #   False → 使用原始属性文本（兼容 P1 checkpoint，无需 GPT 推理结果）
    #   True  → 使用 GPT 增强文本（与 P2 一致，格式由 cfg.product.text_fmt 决定）
    #            此时 P4 仍以 use_gpt=False 的单段 CLS 推理（不做 sliding-window）
    #            以控制训练显存。checkpoint 兼容性靠 strict=False 保证。
    cfg.lm.train.p4_use_gpt = False

    # p4_lm_batch_size（默认 0 = 自动）：P4 联合训练中 LM 推理/训练的批次大小
    #   0  → 自动模式：先根据可用 GPU 显存估算，再以 OOM 自动减半保底
    #   >0 → 固定批次，跳过自动估算（适合已知显存充足时强制提速）
    # 估算公式（fp16 + gradient_checkpointing）：
    #   可用 = free_GB - 2GB headroom; batch ≈ floor(可用 × 16) 取最近 2 的幂
    #   示例：40GB 卡模型加载后余 30GB → batch=32；8GB 卡余 4GB → batch=8
    cfg.lm.train.p4_lm_batch_size = 0  # 0 = auto

    # p4_max_length（默认 0 = 数据集自动）：P4 联合训练时 tokenizer 的 max_length 上限。
    #   DeBERTa disentangled attention 的 p2c_att 张量形状为
    #   [batch×12, seq_len, 2×att_span]（fp32），显存与 seq_len² 成正比。
    #
    #   0  → 按数据集自动选取（见 JointGNNLMTrainer.DATASET_MAX_LENGTH）：
    #          · cora       → 256（原始文本中位数 180 tok，93% 节点 ≤256 tok，
    #                              attention 显存降 4×，仅截断 7% 节点尾部）
    #          · arxiv_2023 → 384（P50=264 tok，P90=371，P95=405；384 仅截断 7.3%，
    #                              (384/512)²≈0.56，节省 ~44% 每次 LM 计算时间）
    #          · 其余数据集  → 512（保持原始行为）
    #   >0 → 强制使用指定值，覆盖数据集默认（适合调试或特殊数据集）
    cfg.lm.train.p4_max_length = 0  # 0 = 数据集自动

    # p4_refresh_interval（默认 0 = 数据集自动）：P4 每隔多少 epoch 刷新全图 LM 嵌入缓存。
    #   非训练节点（约 97% 节点）的 embedding 从缓存读取；
    #   缓存每 p4_refresh_interval epoch 用最新 LM 权重重推理一次。
    #
    #   0  → 按数据集自动选取（见 JointGNNLMTrainer.DATASET_REFRESH_INTERVAL）：
    #          · arxiv_2023 → 20（33868 节点，每次刷新 ~155s；间隔 20 减少 ~6 次刷新
    #                              ≈节省 930s，对最终精度无影响——eval_and_save 总是
    #                              用最新 LM 做最终全图推理）
    #          · 其余数据集  → 10（默认，保持原始行为）
    #   >0 → 强制使用指定值（覆盖数据集默认）
    cfg.lm.train.p4_refresh_interval = 0  # 0 = 数据集自动

    # ── cora 专用 P2 改进方案（仅 dataset=cora 时生效，其他数据集自动忽略） ──
    #
    # 方案A：include_initial_reasoning（默认 False）
    #   【仅对 cora 生效；arxiv_2023 等数据集始终使用双iter（initial+refined），不受此参数影响】
    #   False（推荐）→ cora 只拼接 refined 响应，与 TAPTN 原始做法一致（seeds 0-3 均值 85.15%）
    #   True         → cora 同时拼接 initial + refined（文本增长 +49%，实测降至 ~82-83%，不推荐）
    cfg.lm.train.include_initial_reasoning = False

    # 方案B：label_prefix_in_text（默认 False）
    #   是否将从 GPT 精炼回应中提取的预测类别标签以
    #   "[Predicted Category: {label}]" 形式前置到文本开头。
    #   将标签信号从文本末尾（93% 位置）移至首 chunk → 理论上可超越 TAPTN 基准。
    #   99.9% 节点可成功提取，失败时退化为无前缀（refined-only）。
    cfg.lm.train.label_prefix_in_text = False

    # 方案C：use_gpt_soft_label（默认 False）
    #   是否从 GPT 精炼回应中提取 top-2 预测标签及其评分，
    #   构建归一化软独热向量（FloatTensor 7维），作为 KL 正则项辅助训练。
    #   损失 = CE(logits, hard_labels) + kl_weight × KL(model_prob || gpt_soft)
    #   全零行（提取失败节点）自动跳过，不贡献 KL 损失。
    cfg.lm.train.use_gpt_soft_label = False

    # 方案C 权重：gpt_soft_label_weight（默认 0.3）
    #   KL 损失的权重系数。0.0 → 完全禁用；越大 GPT 软标签影响越强。
    #   推荐范围：0.1～0.5（过大会压制真实标签的 CE 信号）。
    cfg.lm.train.gpt_soft_label_weight = 0.3


    # ------------------------------------------------------------------------ #
    # Actor dataset – multi-channel LLM reasoning options
    # ------------------------------------------------------------------------ #
    cfg.actor = CN()

    # iter1 channel pkl 文件列表（用于 Initial Classification and Reasoning）
    # 与 LLM-Structured-Data-main/run_taptn.py 调试配置保持一致
    cfg.actor.channel_pkls = [
        f'{_PKL_HOME}/actor_hop1_iter1_neighbors_instr_llama_3.3_70b_i_5.pkl',
        f'{_PKL_HOME}/actor_hop1_iter1_neighbors_instr_refine_llama_3.3_70b_i.pkl',
        f'{_PKL_HOME}/actor_hop1_iter1_neighbors_instr_llama_3.3_70b_i_tn1_2.pkl',
        f'{_PKL_HOME}/actor_hop1_iter1_neighbors_llama_3.3_70b_i.pkl',
    ]

    # iter2 pkl 文件（用于 Refined Classification and Reasoning）
    cfg.actor.iter2_pkl = (
        f'{_PKL_HOME}/actor_hop1_iter2_neighbors_llama_3.3_70b_i.pkl'
    )

    # channel pkl 生成时使用的节点集
    cfg.actor.channels_node_set = 'test_and_1hop'

    # 是否在多通道段首显示 Consensus 摘要行
    cfg.actor.show_consensus = False

    return cfg


# Principle means that if an option is defined in a YACS config object,
# then your program should set that configuration option using cfg.merge_from_list(opts) and not by defining,
# for example, --train-scales as a command line argument that is then used to set cfg.TRAIN.SCALES.


def update_cfg(cfg, args_str=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="",
                        metavar="FILE", help="Path to config file")
    # opts arg needs to match set_cfg
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    if isinstance(args_str, str):
        # parse from a string
        args = parser.parse_args(args_str.split())
    else:
        # parse from command line
        args = parser.parse_args()
    # Clone the original cfg
    cfg = cfg.clone()

    # Update from config file
    if os.path.isfile(args.config):
        cfg.merge_from_file(args.config)

    # Update from command line
    cfg.merge_from_list(args.opts)
    if cfg.dataset in ['cora']:
        cfg.runs = 4
    elif cfg.dataset in ['arxiv_2023']:
        cfg.runs = 3
    elif cfg.dataset in ['actor', 'product']:
        cfg.runs = 1
    else:
        cfg.runs = 4
    if cfg.gnn.model.name in ['SAGE','GAT']:
        cfg.gnn.model.hidden_dim = 1024

    return cfg


"""
    Global variable
"""
cfg = set_cfg(CN())
