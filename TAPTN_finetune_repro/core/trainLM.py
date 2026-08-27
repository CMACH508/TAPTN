# ── GPU 选择（必须在任何 torch/CUDA 初始化之前执行）────────────────────────
# 设计原则：
#   HuggingFace Trainer 会对所有"可见 GPU"立即启动 DataParallel，
#   因此不能靠"排序"来实现"主卡满了再用其他卡"，
#   正确做法是：只把恰好够用的最少 GPU 暴露给进程。
#
# 策略：按空闲显存降序贪心选卡，累计空闲量 >= REQUIRED_MEM_MIB 时停止。
#   - 若仅一张卡的空闲量就足够 → 只暴露那一张（其余卡完全不被占用）
#   - 若需要多张卡 → 按空闲量从大到小依次加入，直到总量满足需求
# ────────────────────────────────────────────────────────────────────────────
import os
import subprocess

def _estimate_required_mib() -> int:
    """
    按照 dataset + use_gpt 动态估算 LM 训练所需显存（MiB）。

    优先读取环境变量 LM_REQUIRED_MEM_MIB（手动覆盖）；
    若未设置，则从 sys.argv 解析 dataset / lm.train.use_gpt，
    按下表返回经验值：

        dataset              use_gpt=True   use_gpt=False
        ───────────────────  ─────────────  ─────────────
        product / actor      20 000 MiB     4 000 MiB
          (长文本+slid.win)   (~18 GB)       (~2 GB, 512 trunc)
        arxiv_2023           12 000 MiB     6 000 MiB
        WebKB / cora / etc.   8 000 MiB     4 000 MiB
    """
    import sys
    env_val = os.environ.get('LM_REQUIRED_MEM_MIB', '').strip()
    if env_val.isdigit():
        return int(env_val)

    # 从 yacs 风格的 key-value sys.argv 中快速提取关键参数
    kv: dict = {}
    argv = sys.argv[1:]
    for i in range(0, len(argv) - 1, 2):
        kv[argv[i]] = argv[i + 1]

    dataset = kv.get('dataset', '').lower()
    use_gpt = kv.get('lm.train.use_gpt', 'false').lower() == 'true'

    if dataset in ('product', 'actor'):
        # use_gpt=True → sliding window，最长 ~4000-6000 token，backprop ≈ 18 GB
        # use_gpt=False → truncation=True max_length=512，约 2-3 GB
        return 20_000 if use_gpt else 4_000
    elif dataset == 'arxiv_2023':
        return 12_000 if use_gpt else 6_000
    else:
        # WebKB (wisconsin / texas / cornell), cora 等短文本
        return 8_000 if use_gpt else 4_000


# 在模块加载时（import torch 之前）完成估算，供下方函数使用。
REQUIRED_MEM_MIB = _estimate_required_mib()


def _select_gpus_by_free_memory(required_mib: int = REQUIRED_MEM_MIB):
    """
    从 CUDA_VISIBLE_DEVICES 中贪心选出空闲显存累计 >= required_mib 的最少 GPU，
    并将 CUDA_VISIBLE_DEVICES 更新为仅含这些 GPU（按空闲量降序）。

    必须在 import torch 之前调用，否则 CUDA_VISIBLE_DEVICES 的修改对 CUDA 无效。
    """
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible or visible.strip() == '-1':
        return  # 未指定 GPU 或已禁用 CUDA

    # 解析候选物理 GPU 编号
    gpu_ids = []
    for x in visible.split(','):
        x = x.strip()
        if x.isdigit():
            gpu_ids.append(int(x))

    if not gpu_ids:
        return

    # 通过 nvidia-smi 查询各 GPU 的空闲显存（MiB）
    try:
        result = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=index,memory.free',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print('[GPU selector] nvidia-smi 不可用，保持原始 CUDA_VISIBLE_DEVICES。')
        return

    free_mem = {}
    for line in result.stdout.strip().split('\n'):
        parts = line.strip().split(',')
        if len(parts) == 2:
            try:
                free_mem[int(parts[0].strip())] = int(parts[1].strip())
            except ValueError:
                continue

    # 按空闲显存降序排列候选 GPU
    sorted_ids = sorted(gpu_ids, key=lambda g: free_mem.get(g, 0), reverse=True)

    # 贪心：累计空闲量刚好满足需求时停止，只暴露必要的卡
    selected, cumulative = [], 0
    for g in sorted_ids:
        selected.append(g)
        cumulative += free_mem.get(g, 0)
        if cumulative >= required_mib:
            break
    # 若所有卡加起来仍不足，保留全部（让运行时自行报 OOM）

    new_visible = ','.join(str(g) for g in selected)
    os.environ['CUDA_VISIBLE_DEVICES'] = new_visible

    print(f'[GPU selector] 预估所需显存: {required_mib} MiB'
          f'  (可通过环境变量 LM_REQUIRED_MEM_MIB 调整)')
    print('[GPU selector] 候选 GPU 空闲显存 (MiB):')
    for g in sorted_ids:
        chosen = '✓ 已选用' if g in selected else '✗ 保留空闲'
        print(f'  物理 GPU {g}: {free_mem.get(g, "?")} MiB  [{chosen}]')
    print(f'[GPU selector] CUDA_VISIBLE_DEVICES: {visible!r} → {new_visible!r}')
    print(f'[GPU selector] 共选 {len(selected)} 张卡，累计空闲 {cumulative} MiB')


_select_gpus_by_free_memory()
# ── 以下 import 会触发 torch/CUDA 初始化，须在上方排序完成后执行 ────────────

from core.LMs.lm_trainer import LMTrainer
from core.config import cfg, update_cfg
import pandas as pd


from core.utils import lm_ckpt_stem


def _ckpt_path_for(cfg, seed: int) -> str:
    """推断给定 seed 的 LM checkpoint 路径（与 LMTrainer 中的 ckpt_dir 保持一致）。"""
    use_gpt = cfg.lm.train.use_gpt
    sem     = getattr(cfg.lm.train, 'sem', True)
    if use_gpt:
        suffix = '3' if sem else '3nosem'
    else:
        suffix = ''
    return f'{lm_ckpt_stem(cfg, cfg.dataset, suffix, seed)}.ckpt'


def run(cfg):
    seeds        = [cfg.seed] if cfg.seed is not None else range(cfg.runs)
    force_retrain = getattr(cfg.lm.train, 'force_retrain', False)
    all_acc = []

    for seed in seeds:
        cfg.seed = seed

        ckpt_path = _ckpt_path_for(cfg, seed)
        ckpt_exists = os.path.exists(ckpt_path)

        if ckpt_exists and not force_retrain:
            # ── 跳过训练，直接加载已有 checkpoint 做推理 ──────────────────
            print(f'\n[trainLM] ⚡ 发现已有 checkpoint: {ckpt_path}')
            print(f'[trainLM]    force_retrain=False → 跳过训练，仅重新生成 .emb/.pred')
            trainer = LMTrainer(cfg)
            # 将已有权重加载到模型（确保 eval_and_save 使用微调后的参数）
            import torch
            state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            # strict=False：允许旧 checkpoint 缺少新增模块（如 chunk_attn_pool）
            # 缺失权重保持随机初始化，推理时可能精度偏低，此处打印警告
            load_result = trainer.model.load_state_dict(state, strict=False)
            if load_result.missing_keys:
                print(f'[trainLM] ⚠️  checkpoint 缺少以下权重（将使用随机初始化）: '
                      f'{load_result.missing_keys}')
                print(f'[trainLM]    提示：该 checkpoint 与当前模型配置不匹配，'
                      f'建议使用 --force-retrain 重新训练。')
            if load_result.unexpected_keys:
                print(f'[trainLM] ⚠️  checkpoint 含多余权重（已忽略）: '
                      f'{load_result.unexpected_keys}')
            print(f'[trainLM]    已加载 checkpoint 权重: {ckpt_path}')
        else:
            if ckpt_exists and force_retrain:
                print(f'\n[trainLM] 🔄 force_retrain=True → 忽略已有 checkpoint，从 DeBERTa 重新微调')
                print(f'[trainLM]    将覆盖: {ckpt_path}')
            trainer = LMTrainer(cfg)
            trainer.train()

        acc = trainer.eval_and_save()
        all_acc.append(acc)

    if len(all_acc) > 1:
        df = pd.DataFrame(all_acc)
        for k, v in df.items():
            print(f"{k}: {v.mean():.4f} ± {v.std():.4f}")


if __name__ == '__main__':
    cfg = update_cfg(cfg)
    run(cfg)
