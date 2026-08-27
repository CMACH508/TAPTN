"""
core/trainJoint.py  —  P4 流水线入口

在 GNN 训练阶段同时微调 LM（联合训练）。
LM 从 P1 checkpoint 初始化；训练结束后只保存 GNN，不保存 LM。

用法示例（命令行）：
    python -m core.trainJoint \\
        dataset wisconsin \\
        seed 1 \\
        gnn.model.name GATv2 \\
        device 0

不支持 RevGAT（依赖 DGL），遇到时自动跳过并打印警告。
"""

from core.config import cfg, update_cfg
from core.GNNs.joint_trainer import JointGNNLMTrainer
import pandas as pd
import time


def run(cfg):
    seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)

    if cfg.gnn.model.name == "RevGAT":
        print(
            "[P4] 跳过 RevGAT：JointGNNLMTrainer 不支持 DGL 后端，"
            "请改用其他 GNN 模型。"
        )
        return

    all_acc = []
    start   = time.time()

    for seed in seeds:
        cfg.seed = seed
        trainer  = JointGNNLMTrainer(cfg, feature_type="TA")
        trainer.train()
        _, acc = trainer.eval_and_save()
        all_acc.append(acc)

    end = time.time()

    if len(all_acc) > 1:
        df = pd.DataFrame(all_acc)
        print(
            f"[{cfg.gnn.model.name} + LM-joint(P4)] "
            f"ValACC: {df['val_acc'].mean():.4f} ± {df['val_acc'].std():.4f}, "
            f"TestAcc: {df['test_acc'].mean():.4f} ± {df['test_acc'].std():.4f}"
        )

    print(f"Running time: {(end - start) / len(seeds):.2f}s")


if __name__ == "__main__":
    cfg = update_cfg(cfg)
    run(cfg)
