#!/usr/bin/env python3
"""
TAPTN (Text-Attributed Prompt-based Topology-aware Node classification) 命令行工具

使用示例:
    # 第一次迭代
    python run_taptn.py --dataset cora --api_key YOUR_KEY --base_url YOUR_URL
    
    # 第二次迭代（精炼）
    python run_taptn.py --dataset cora --api_key YOUR_KEY --base_url YOUR_URL \\
        --iteration 2 --prev_results results_iter1.pkl --prev_reasons reasons_iter1.pkl
    
    # 使用配置文件
    python run_taptn.py --config config.json
"""

import argparse
import json
import pickle
import sys
import os

_VENDOR = os.path.dirname(os.path.abspath(__file__))
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)
import numpy as np
from datetime import datetime

from utils.utils import process_and_compare_predictions, load_data, sample_test_nodes, map_arxiv_labels, build_multichannel_judgement
from utils.prompts import arxiv_natural_lang_mapping


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='TAPTN: 基于LLM的图节点分类工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用
  %(prog)s --dataset cora --api_key YOUR_KEY --base_url YOUR_URL
  
  # 完整 TAPTN 流程（第一次迭代）
  %(prog)s --dataset cora --api_key YOUR_KEY --base_url YOUR_URL \\
      --mode neighbors --hop 1 --use_instructions --first_iter
  
  # 第二次迭代（精炼）
  %(prog)s --dataset cora --api_key YOUR_KEY --base_url YOUR_URL \\
      --iteration 2 --prev_results results.pkl --prev_reasons reasons.pkl \\
      --refining --use_instructions
  
  # 使用配置文件
  %(prog)s --config config.json
  
  # 2跳邻居分类
  %(prog)s --dataset cora --api_key YOUR_KEY --base_url YOUR_URL \\
      --hop 2 --max_papers_1 20 --max_papers_2 5
        """
    )
    
    # 配置文件
    parser.add_argument('--config', type=str, help='JSON配置文件路径')
    
    # 基本参数
    parser.add_argument('--dataset', type=str, default='actor',
                       choices=['cora', 'arxiv_2023', 'texas', 'wisconsin', 'washington', 'cornell', 'actor'],
                       help='数据集名称 (默认: cora)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42)')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='输出目录 (默认: ./results)')
    
    # API 配置
    parser.add_argument('--api_key', type=str, required=False, default='',
                       help='LLM API key (or set OPENAI_API_KEY / TAPTN_LLM_API_KEY)')
    parser.add_argument('--base_url', type=str, required=False, default='',
                       help='LLM API base URL (or set OPENAI_BASE_URL)')
    parser.add_argument('--model', type=str, default='meta-llama/llama-3.3-70b-instruct',
                       help='LLM模型名称 (默认: gpt-4o)')
    
    # 性能参数
    parser.add_argument('--max_workers', type=int, default=200,
                       help='最大并发工作线程数 (默认: 1 串行处理，推荐: 5-10 并发处理)')
    parser.add_argument('--timeout', type=int, default=6000,
                       help='单个节点处理超时时间(秒) (默认: 6000)')
    
    # 图结构参数
    parser.add_argument('--mode', type=str, default='neighbors',
                       choices=['ego', 'neighbors'],
                       help='分类模式: ego(仅自身) 或 neighbors(考虑邻居) (默认: neighbors)')
    parser.add_argument('--hop', type=int, default=1, choices=[1, 2],
                       help='邻居跳数 (默认: 1)')
    parser.add_argument('--max_papers_1', type=int, default=20,
                       help='每类最多采样的1跳邻居数 (默认: 20)')
    parser.add_argument('--max_papers_2', type=int, default=5,
                       help='每类最多采样的2跳邻居数 (默认: 5)')
    
    # 迭代参数
    parser.add_argument('--iteration', type=int, default=2, choices=[1, 2],
                       help='迭代次数: 1(首次) 或 2(精炼) (默认: 1)')
    parser.add_argument('--prev_results', type=str,default='cot_results_cora_hop1_iter1_neighbors_instr_llama_3.3_70b_i_all_1.pkl',
                       help='前一次迭代的结果文件(.pkl)')
    parser.add_argument('--prev_reasons', type=str, default='cora_hop1_iter1_neighbors_instr_llama_3.3_70b_i_all_1.pkl',
                       help='前一次迭代的推理文件(.pkl)')
    parser.add_argument('--channel_pkl', dest='channel_pkl', action='append',
                       metavar='PKL', default=None,
                       help='多 channel 模式：可重复指定，每次添加一个 channel pkl 文件。'
                            '设置后将对每个节点合并所有 channel 的推理（用于 Actor iter2）。'
                            '例如: --channel_pkl a.pkl --channel_pkl b.pkl --channel_pkl c.pkl')
    parser.add_argument('--prev_reasons_channels', type=str, default=None,
                       help='多 channel 模式（兼容旧格式）：逗号或空格分隔的多个 iter1 pkl 路径。'
                            '例如: "a.pkl,b.pkl,c.pkl,d.pkl"（推荐使用 --channel_pkl 代替）')
    parser.add_argument('--channels_node_set', type=str, default='test_and_1hop',
                       help='各 channel pkl 生成时使用的节点集 (默认: test_and_1hop)')
    parser.add_argument('--show_consensus', action='store_true', default=False,
                       help='多 channel 模式下，在各 channel 推理前插入共识摘要行 (默认: 关闭)')
    parser.add_argument('--refining', action='store_true', default=False,
                       help='使用精炼策略 (仅第2次迭代)')
    parser.add_argument('--use_instructions', action='store_true', 
                       help='使用TAPTN分步指导')
    parser.add_argument('--just_reflection', action='store_true',
                       help='仅使用自我反思策略')
    parser.add_argument('--rewiring', action='store_true', default=False,
                       help='使用图重连策略')
    
    # Prompt 参数
    parser.add_argument('--zero_shot_cot', action='store_true', default=True,
                       help='使用零样本思维链 (默认: True)')
    parser.add_argument('--few_shot', action='store_true',
                       help='使用少样本学习')
    parser.add_argument('--include_abs', action='store_true', default=True,
                       help='包含摘要信息 (默认: True)')
    parser.add_argument('--include_label', action='store_true',
                       help='在邻居信息中包含标签')
    parser.add_argument('--include_options', action='store_true',
                       help='在prompt中明确列出类别选项')
    parser.add_argument('--anonymize_edges', action='store_true', default=False,
                       help='匿名化边')
    parser.add_argument('--webkb_full_abs', action='store_true', default=False,
                       help='(仅WebKB数据集有效) 对目标节点及所有邻居节点均显示完整内容摘要；'
                            '默认关闭时仅对 "other" 类型邻居附加摘要。')
    parser.add_argument('--include_neighbors', action='store_true', default=False,
                       help='(仅WebKB数据集 + hop=2 有效) 在每个一阶邻居的描述中附加其二阶链接模式统计；'
                            '默认关闭，关闭时即使 hop=2 也不附加该信息。')
    # ArXiv 特定参数
    parser.add_argument('--arxiv_style', type=str, default='subcategory',
                       choices=['subcategory', 'identifier', 'natural language'],
                       help='ArXiv标签风格 (默认: subcategory)')
    
    # 采样参数
    parser.add_argument('--sample_size', type=int, default=None,
                       help='节点采样大小 (默认: None, 使用全部可用节点，无上限)')
    parser.add_argument('--node_set', type=str, default='test',
                       choices=['test', 'train', 'val', 'all', 'train+val', 'val+test',
                                'test_has_test_nbr', 'test_and_1hop'],
                       help='选择要测试的节点集 (默认: test)\n' + 
                            '  test: 仅测试集 (Actor: 889节点)\n' +
                            '  train: 仅训练集 (Actor: 2116节点)\n' +
                            '  val: 仅验证集 (Actor: 1411节点)\n' +
                            '  all: 所有节点 (Actor: 4416节点)\n' +
                            '  train+val: 训练+验证集\n' +
                            '  val+test: 验证+测试集\n' +
                            '  test_has_test_nbr: 测试集中1阶邻居含至少一个测试集节点的子集\n' +
                            '  test_and_1hop: 测试集节点 ∪ 其所有1阶邻居（含非测试集节点）')
    parser.add_argument('--explain', action='store_true',
                       help='输出详细解释信息')
    
    return parser.parse_args()


def load_config(config_path):
    """从JSON文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def merge_config(args, config):
    """合并命令行参数和配置文件"""
    for key, value in config.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)
    return args


def validate_args(args):
    """验证参数"""
    errors = []
    
    # 检查必需参数
    if not args.api_key and not (os.environ.get("OPENAI_API_KEY") or os.environ.get("TAPTN_LLM_API_KEY")):
        errors.append("missing --api_key (or OPENAI_API_KEY / TAPTN_LLM_API_KEY)")
    
    # 检查迭代参数
    if args.iteration == 2:
        # 解析 channel 列表（与 main 逻辑保持一致）
        _ch = list(args.channel_pkl) if args.channel_pkl else []
        if not _ch and args.prev_reasons_channels:
            _ch = [p.strip() for part in args.prev_reasons_channels.split(',')
                   for p in part.split() if p.strip()]
        if _ch:
            # 多 channel 模式：验证每个文件存在
            for ch_pkl in _ch:
                if not os.path.exists(ch_pkl):
                    errors.append(f"Channel pkl 文件不存在: {ch_pkl}")
        else:
            if not args.prev_results or not args.prev_reasons:
                errors.append("第2次迭代需要 --prev_results 和 --prev_reasons 参数")
            if args.prev_results and not os.path.exists(args.prev_results):
                errors.append(f"前一次迭代结果文件不存在: {args.prev_results}")
            if args.prev_reasons and not os.path.exists(args.prev_reasons):
                errors.append(f"前一次迭代推理文件不存在: {args.prev_reasons}")
    
    if errors:
        print("参数验证失败:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)


def setup_output_dir(output_dir, dataset_name):
    """创建输出目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, dataset_name, timestamp)
    os.makedirs(output_path, exist_ok=True)
    return output_path


def save_config(args, output_path):
    """保存运行配置"""
    config_dict = vars(args).copy()
    config_path = os.path.join(output_path, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    print(f"配置已保存到: {config_path}")


def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置文件（如果提供）
    if args.config:
        config = load_config(args.config)
        args = merge_config(args, config)
    
    # 验证参数
    validate_args(args)
    
    # 创建输出目录
    output_path = setup_output_dir(args.output_dir, args.dataset)
    print(f"\n输出目录: {output_path}\n")
    
    # 保存配置
    save_config(args, output_path)
    
    # 确定数据源
    if args.dataset == "arxiv_2023":
        source = "arxiv"
    elif args.dataset in ["texas", "wisconsin", "washington", "cornell"]:
        source = "wisconsin"
    elif args.dataset == "actor":
        source = "actor"
    else:
        source = args.dataset
    
    print("=" * 80)
    print(f"TAPTN 节点分类")
    print("=" * 80)
    print(f"数据集: {args.dataset}")
    print(f"模式: {args.mode}")
    print(f"跳数: {args.hop}")
    print(f"迭代: 第 {args.iteration} 次")
    print(f"使用指导: {args.use_instructions}")
    print(f"精炼: {args.refining}")
    print(f"模型: {args.model}")
    print("=" * 80 + "\n")
    
    # 加载数据
    print("加载数据...")
    data, data2, text = load_data(args.dataset, use_text=True, seed=args.seed)
    print(f"✓ 数据加载完成")
    print(f"  节点数: {data.num_nodes}")
    print(f"  边数: {data.edge_index.shape[1]}")
    print(f"  训练集: {data.train_mask.sum().item()}")
    print(f"  验证集: {data.val_mask.sum().item()}")
    print(f"  测试集: {data.test_mask.sum().item()}\n")
    
    # 处理 ArXiv 标签
    if source == "arxiv" and args.arxiv_style != "subcategory":
        text = map_arxiv_labels(data, text, source, args.arxiv_style)
    
    # 设置类别选项（需在加载 initial_judgements 之前，因多 channel 模式需要 options 做 majority vote）
    options = set(text['label'])
    if args.dataset == "arxiv_2023":
        options = set(['cs.GT', 'cs.MA', 'cs.RO', 'cs.NE', 'cs.IR', 'cs.SI', 'cs.CY'])
    elif args.dataset == "cora":
        if "Theory" in options:
            options.remove("Theory")
            options.add("Computational Learning Theory")
    elif args.dataset == "actor":
        pass  # Actor 类别已在 load_actor.py 中正确设置

    # 加载先前迭代的结果
    initial_judgements = None
    if args.iteration == 2:
        # ── 解析多 channel 文件列表 ──────────────────────────────────
        # 优先：--channel_pkl（可重复指定，每次一个文件）
        channel_pkls = list(args.channel_pkl) if args.channel_pkl else []
        # 兼容旧格式：--prev_reasons_channels（逗号/空格分隔）
        if not channel_pkls and args.prev_reasons_channels:
            raw = args.prev_reasons_channels
            # 先按逗号分割，再按空格分割，去除空项
            channel_pkls = [p.strip() for part in raw.split(',')
                            for p in part.split() if p.strip()]

        if channel_pkls:
            # ── 多 channel 模式 ──────────────────────────────────────
            print(f"加载多 channel 初始推理（{len(channel_pkls)} 个 channel）...")
            for i, p in enumerate(channel_pkls):
                print(f"  Channel {i+1}: {p}")
            initial_judgements = build_multichannel_judgement(
                channel_pkls, data, text,
                dataset=args.dataset,
                node_set=args.channels_node_set,
                options=options,
                show_consensus=args.show_consensus,
            )
            print("✓ 多 channel 初始推理加载完成\n")
        else:
            # ── 单 channel 模式（原有逻辑）──────────────────────────────
            print("加载前一次迭代的结果...")
            initial_results = pickle.load(open(args.prev_results, "rb"))
            initial_judgements_data = pickle.load(open(args.prev_reasons, "rb"))['wrong_reason']
            initial_judgements = [initial_judgements_data, initial_results]
            print("✓ 前一次迭代结果加载完成\n")
    
    # 采样节点
    # 根据 node_set 确定可用节点数
    if args.node_set == 'test':
        available_count = len(data.test_id)
    elif args.node_set == 'train':
        available_count = len(data.train_id)
    elif args.node_set == 'val':
        available_count = len(data.val_id)
    elif args.node_set == 'all':
        available_count = data.num_nodes
    elif args.node_set == 'train+val':
        available_count = len(data.train_id) + len(data.val_id)
    elif args.node_set == 'val+test':
        available_count = len(data.val_id) + len(data.test_id)
    elif args.node_set == 'test_has_test_nbr':
        # 预估：实际数量由 sample_test_nodes 计算，这里用测试集总数作上界
        available_count = len(data.test_id)
    elif args.node_set == 'test_and_1hop':
        # 预估：实际数量由 sample_test_nodes 计算，这里用全图节点数作上界
        available_count = data.num_nodes
    
    # 使用用户指定的sample_size，如果为None则使用所有可用节点
    sample_size = args.sample_size  # 不再使用min(1000, available_count)限制
    if sample_size is None:
        print(f"从 '{args.node_set}' 集合预测所有节点 (可用: {available_count})...")
    else:
        print(f"从 '{args.node_set}' 集合采样 {sample_size} 个节点 (可用: {available_count})...")
    node_indices = sample_test_nodes(data, text, sample_size, args.dataset, node_set=args.node_set)
    node_index_list = list(node_indices)
    print(f"✓ 处理完成: {len(node_index_list)} 个节点\n")
    
    # 设置 first_iter
    first_iter = (args.iteration == 1)
    
    # 运行分类
    print("开始节点分类...")
    print("-" * 80)
    accuracy, wrong_indexes_list = process_and_compare_predictions(
        node_index_list, 
        data, 
        text, 
        dataset_name=args.dataset,
        source=source,
        mode=args.mode,
        max_papers_1=args.max_papers_1,
        max_papers_2=args.max_papers_2,
        hop=args.hop,
        include_label=args.include_label,
        arxiv_style=args.arxiv_style,
        include_abs=args.include_abs,
        include_options=args.include_options,
        zero_shot_CoT=args.zero_shot_cot,
        few_shot=args.few_shot,
        options=options,
        initial_judgement=initial_judgements,
        explain=args.explain,
        api_key=args.api_key,
        base_url=args.base_url,
        refining=args.refining,
        just_reflection=args.just_reflection,
        use_instructions=args.use_instructions,
        first_iter=first_iter,
        rewiring=args.rewiring,
        model=args.model,
        timeout=args.timeout,
        max_workers=args.max_workers,
        anonymize_edges=args.anonymize_edges,
        webkb_full_abs=args.webkb_full_abs,
        include_neighbors=args.include_neighbors
    )
    print("-" * 80)
    
    # 输出结果
    print("\n" + "=" * 80)
    print("分类结果")
    print("=" * 80)
    print(f"准确率: {accuracy:.2%}")
    print(f"正确数: {int(accuracy * len(node_index_list))}/{len(node_index_list)}")
    print(f"错误数: {len(wrong_indexes_list)}")
    print("=" * 80 + "\n")
    
    # 保存结果摘要
    summary = {
        'dataset': args.dataset,
        'mode': args.mode,
        'hop': args.hop,
        'iteration': args.iteration,
        'use_instructions': args.use_instructions,
        'refining': args.refining,
        'accuracy': float(accuracy),
        'correct': int(accuracy * len(node_index_list)),
        'total': len(node_index_list),
        'wrong_count': len(wrong_indexes_list),
        'timestamp': datetime.now().isoformat()
    }
    
    summary_path = os.path.join(output_path, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"结果摘要已保存到: {summary_path}")
    
    # 保存错误索引
    if wrong_indexes_list:
        wrong_path = os.path.join(output_path, 'wrong_indexes.pkl')
        pickle.dump(wrong_indexes_list, open(wrong_path, 'wb'))
        print(f"错误索引已保存到: {wrong_path}")
    
    print(f"\n所有结果已保存到: {output_path}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

