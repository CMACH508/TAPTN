"""
TAPTN 扩充实验运行器（E1 第二层：新一代开源模型上的 GraphICL vs TAPTN）

复刻论文 Method 表的四个配置（均为非匿名/结构感知模板，参数与
run_missing_experiments.py / main.py 的历史 WebKB·Cora 运行完全一致）：
  graphicl1 : hop=1, 无指令              （GraphICL 1-hop）
  taptn1    : hop=1, 有指令              （TAPTN 1-hop）
  graphicl2 : hop=2, 无指令              （GraphICL 2-hop）
  taptn2    : hop=1, 有指令 + 以 taptn1 结果为 initial judgement（TAPTN 2-hop / iter2）

结果自动保存为 {dataset}_hop{h}_noanon_{guide|noguide}_{model}.pkl；
taptn2 完成后输出重命名为 ..._guide_{model}_iter2.pkl，并恢复 iter1 文件。

用法:
  python run_taptn_expansion.py --dataset texas --model gemma-4-31b-it --config taptn1
"""
import argparse
import os
import pickle
import shutil

import numpy as np
import torch

from utils.utils import process_and_compare_predictions, load_data

MODEL_MAP = {
    "gemma-4-31b-it": "google/gemma-4-31b-it",
    "qwen3.5-27b":    "qwen/qwen3.5-27b",
    "gpt-oss-120b":   "openai/gpt-oss-120b",
    "glm-5.1":        "z-ai/glm-5.1",
    "llama-3.3-70b":  "meta-llama/llama-3.3-70b-instruct",
}

CONFIGS = {
    "graphicl1": dict(hop=1, use_instructions=False, iter2=False),
    "taptn1":    dict(hop=1, use_instructions=True,  iter2=False),
    "graphicl2": dict(hop=2, use_instructions=False, iter2=False),
    "taptn2":    dict(hop=1, use_instructions=True,  iter2=True),
    # 0-hop 纯文本基线：仅目标节点 Content Abstract + URL，无任何邻居/结构信息
    "ego":       dict(hop=1, use_instructions=False, iter2=False, mode="ego"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["texas", "cornell", "washington", "wisconsin", "cora"])
    parser.add_argument("--model", required=True, choices=list(MODEL_MAP.keys()))
    parser.add_argument("--config", required=True, choices=list(CONFIGS.keys()))
    parser.add_argument("--max_workers", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0,
                        help="仅取前 N 个测试节点（冒烟测试用，0=全部）")
    parser.add_argument("--instr_v2", action="store_true",
                        help="使用修订版指令（Cora=refining3_v2；wisconsin=v3）；结果文件加 _v2 后缀")
    parser.add_argument("--wis_ver", choices=["", "v2", "v3", "2hop", "v2_2hop"], default="",
                        help="wisconsin 指令版本显式选择 v2/v3/2hop/v2_2hop（优先于 --instr_v2）；结果文件加 _{ver} 后缀")
    parser.add_argument("--iter1_suffix", default=None,
                        help="iter2 时显式指定加载哪个 1-hop 文件作为初判来源的后缀（如 _v2）；缺省=随 --wis_ver。用于 2hop 指令复用 v2 的 1-hop 初判")
    parser.add_argument("--anon", action="store_true",
                        help="严格匿名边（边方向语义统一替换为 neighbor）；输出文件名含 _anon_")
    parser.add_argument("--flip", action="store_true",
                        help="flipping 结构消融：整图翻转边方向（data.edge_index[[1,0]]，"
                             "等价 Sec 2 的 1-hop flipping），邻居文本/数量不变、仅 in/out 角色互换；"
                             "输出文件名加 _flip 后缀。WebKB only（与历史 utils.py L1612 一致）")
    parser.add_argument("--intervene", choices=["none", "agnostic", "era"], default="none",
                        help="Cora Stage-1 思维干预：agnostic=纯推理脚手架；era=叠加时代分类法校准；"
                             "none=无干预（等价 noguide 基线）。非 none 时输出文件名含 _intv-{mode}_")
    args = parser.parse_args()

    if args.wis_ver:
        os.environ["TAPTN_INSTR_V2"] = "1"
        os.environ["WISCONSIN_VER"] = args.wis_ver
    elif args.instr_v2:
        os.environ["TAPTN_INSTR_V2"] = "1"

    cfg = dict(CONFIGS[args.config])
    # 干预模式：强制注入指令槽（use_instructions=True），并以环境变量选择干预文本
    if args.intervene != "none":
        os.environ["CORA_INTERVENE"] = args.intervene
        cfg["use_instructions"] = True
    model_name = MODEL_MAP[args.model]
    model_tag = model_name.split("/")[-1].replace(":", "_")

    dataset_name = args.dataset
    # WebKB 四校共用 wisconsin 的提示词模板与数据处理分支
    source = "wisconsin" if dataset_name in ("texas", "cornell", "washington", "wisconsin") else dataset_name

    data, data2, text = load_data(dataset_name, use_text=True, seed=42)
    print(data)

    # ── flipping 结构消融：整图翻转边方向（仅 WebKB，hop=1 配置）─────────────────
    # 与工作区 utils.py 历史实现完全一致：只交换 edge_index 两行，不动 in/out degree，
    # 使每个邻居的 in/out 角色互换、文本与数量保持不变（信息守恒、仅结构错误）。
    if args.flip:
        if dataset_name in ("texas", "cornell", "washington", "wisconsin"):
            data.edge_index = data.edge_index[[1, 0], :]
            print("[flip] edge_index direction reversed (in/out roles swapped)")
        else:
            print(f"[flip] WARNING: --flip 仅对 WebKB 生效，{dataset_name} 已忽略")

    options = set(text["label"])
    if dataset_name == "cora":
        options.discard("Theory")
        options.add("Computational Learning Theory")

    node_indices = np.where(data.test_mask.numpy())[0]
    data.test_id = torch.where(data.test_mask)[0]
    node_index_list = [int(x) for x in node_indices]
    if args.limit > 0:
        node_index_list = node_index_list[:args.limit]
    print(f"[{dataset_name}] test nodes: {len(node_index_list)}")

    # 自动保存的目标文件名（process_and_compare_predictions 的命名规则）
    _instr_tag = "guide" if cfg["use_instructions"] else "noguide"
    _anon_tag = "anon" if args.anon else "noanon"
    save_path = f"{dataset_name}_hop{cfg['hop']}_{_anon_tag}_{_instr_tag}_{model_tag}.pkl"
    v2_suffix = f"_{args.wis_ver}" if args.wis_ver else ("_v2" if args.instr_v2 else "")
    flip_suffix = "_flip" if args.flip else ""

    # ── iter2：加载 taptn1 的结果作为 initial judgement ────────────────────────
    # iter1 来源须与本次结构条件一致：directed→noanon、edge-blind→anon、flip→_flip。
    initial_judgements = None
    _iter1_suffix = args.iter1_suffix if args.iter1_suffix is not None else v2_suffix
    iter1_path = f"{dataset_name}_hop1_{_anon_tag}_guide_{model_tag}{_iter1_suffix}{flip_suffix}.pkl"
    if cfg["iter2"]:
        if not os.path.exists(iter1_path):
            raise FileNotFoundError(f"taptn2 需要先完成 taptn1：缺少 {iter1_path}")
        _iter1 = pickle.load(open(iter1_path, "rb"))
        reason = dict(_iter1["reason"])
        result = dict(_iter1["result"])
        for _nid in node_index_list:
            reason.setdefault(_nid, "")
            result.setdefault(_nid, "")
        initial_judgements = [reason, result, reason, result]

    # 运行会覆盖 save_path 同名文件（如 v1 taptn1 结果），先备份
    save_backup = None
    if os.path.exists(save_path):
        save_backup = save_path + ".prevbak"
        shutil.copyfile(save_path, save_backup)

    print("=" * 60)
    print(f"Run: {dataset_name} | {args.config} | hop={cfg['hop']} | "
          f"instructions={cfg['use_instructions']} | model={model_name}")
    print("=" * 60)

    accuracy, wrong = process_and_compare_predictions(
        node_index_list, data, text,
        dataset_name=dataset_name,
        source=source,
        mode=cfg.get("mode", "neighbors"),
        hop=cfg["hop"],
        max_papers_1=20,
        max_papers_2=5,
        include_label=False,
        arxiv_style="subcategory",
        include_abs=True,
        include_options=False,
        zero_shot_CoT=True,
        few_shot=False,
        options=options,
        initial_judgement=initial_judgements,
        explain=False,
        anonymize_edges=args.anon,
        use_instructions=cfg["use_instructions"],
        model_name=model_name,
        max_workers=args.max_workers,
    )

    # ── 后处理：iter2/v2 输出改名，恢复被覆盖的旧文件 ──────────────────────────
    final_path = save_path
    if args.intervene != "none":
        final_path = f"{dataset_name}_hop{cfg['hop']}_{_anon_tag}_intv-{args.intervene}_{model_tag}{flip_suffix}.pkl"
    elif cfg["iter2"]:
        final_path = f"{dataset_name}_hop1_{_anon_tag}_guide_{model_tag}_iter2{v2_suffix}{flip_suffix}.pkl"
    elif cfg.get("mode") == "ego":
        final_path = f"{dataset_name}_ego_noguide_{model_tag}{v2_suffix}{flip_suffix}.pkl"
    elif v2_suffix or flip_suffix:
        final_path = save_path.replace(".pkl", f"{v2_suffix}{flip_suffix}.pkl")
    if final_path != save_path:
        if os.path.exists(save_path):
            os.replace(save_path, final_path)
            print(f"[rename] {save_path} -> {final_path}")
        if save_backup and os.path.exists(save_backup):
            os.replace(save_backup, save_path)
            print(f"[restore] 原文件已恢复: {save_path}")
    elif save_backup and os.path.exists(save_backup):
        os.remove(save_backup)  # 原地重跑：新结果即终版，丢弃备份

    print(f"\n[DONE] {dataset_name} | {args.config} | {args.model} | "
          f"Accuracy: {accuracy * 100:.2f}% | wrong={len(wrong)}")


if __name__ == "__main__":
    main()
