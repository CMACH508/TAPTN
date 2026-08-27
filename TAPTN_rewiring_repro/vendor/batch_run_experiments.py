#!/usr/bin/env python3
"""
批量运行 TAPTN 实验脚本（两层并发架构）

自动化运行多个模型在多个数据集上的实验，支持两层并发执行和自动文件重命名。

并发架构:
    第1层 (批量层): 同时运行多个实验 (默认20个)
    第2层 (实验层): 每个实验内并发处理节点 (默认20线程)

使用示例:
    # 运行 rewiring 实验（默认配置：批量20并发 + 实验20并发）
    python batch_run_experiments.py --rewiring
    
    # 运行 no_rewiring 实验
    python batch_run_experiments.py --no-rewiring
    
    # 自定义两层并发数
    python batch_run_experiments.py --rewiring --batch_workers 10 --max_workers 30
    
    # 仅运行特定模型
    python batch_run_experiments.py --rewiring --models phi-4 gemma-2-9b-it
    
    # 仅运行特定数据集
    python batch_run_experiments.py --rewiring --datasets cornell texas
"""

import subprocess
import argparse
import os
import sys
import glob
import time
from datetime import datetime
from pathlib import Path
import concurrent.futures

# 模型列表（与论文Section 2一致）
MODELS = {
    'phi-4': 'microsoft/phi-4',
    'gemma-2-9b-it': 'google/gemma-2-9b-it',
    'llama-3-70b-instruct': 'meta-llama/llama-3-70b-instruct',
    'llama-3.1-70b-instruct': 'meta-llama/llama-3.1-70b-instruct',
    'llama-3.3-70b-instruct': 'meta-llama/llama-3.3-70b-instruct',
    'qwen2.5-vl-72b-instruct': 'qwen/qwen2.5-vl-72b-instruct',
    'gpt-3.5-turbo-0125': 'gpt-3.5-turbo-0125',
    'qwen-2.5-72b-instruct': 'qwen/qwen-2.5-72b-instruct',
    # 2026-06 新模型面板（Reviewer 7rnE，E1）：非 thinking 模式，见 utils/Agent.py
    'gemma-4-31b-it': 'google/gemma-4-31b-it',
    'qwen3.5-27b': 'qwen/qwen3.5-27b',
    'gpt-oss-120b': 'openai/gpt-oss-120b',
    'glm-5.1': 'z-ai/glm-5.1',
}

# 数据集列表（WebKB低同质性数据集）
DATASETS = ['cornell', 'texas', 'wisconsin', 'washington']

def get_expected_pickle_filename(dataset, hop, iteration, mode, use_instructions,
                                  refining, model_full_name, just_reflection=False,
                                  anonymize_edges=False, webkb_full_abs=False):
    """
    根据参数生成预期的 pickle 文件名，与 utils/utils.py 中的
    generate_pickle_filename 保持完全一致。

    命名格式:
        {dataset}_hop{hop}_iter{iter}_{mode}
        [_instr][_refl][_refine][_anon][_full_abs]
        [_{model_short}].pkl

    model_short 计算方式: model_full_name.split('/')[-1].replace('-','_')[:15]
    """
    filename_parts = [
        dataset,
        f"hop{hop}",
        f"iter{iteration}",
        mode,
    ]

    flags = []
    if use_instructions:
        flags.append("instr")
    if just_reflection:
        flags.append("refl")
    if refining:
        flags.append("refine")
    if anonymize_edges:
        flags.append("anon")
    if webkb_full_abs:
        flags.append("full_abs")
    if flags:
        filename_parts.extend(flags)

    if model_full_name:
        model_short = model_full_name.split('/')[-1].replace('-', '_')[:15]
        filename_parts.append(model_short)

    return "_".join(filename_parts) + ".pkl"


def rename_pickle_files(dataset, hop, iteration, mode, use_instructions,
                        refining, model_full_name, rewiring_flag,
                        just_reflection=False, anonymize_edges=False,
                        webkb_full_abs=False):
    """
    重命名生成的 pickle 文件，添加 rewiring 后缀。

    按照 generate_pickle_filename 的命名规则定位文件，依次在
    当前目录和 result2/ 子目录中查找。

    Args:
        dataset: 数据集名称
        hop: 跳数
        iteration: 迭代次数
        mode: 模式 (ego/neighbors)
        use_instructions: 是否使用指导
        refining: 是否使用精炼
        model_full_name: 完整模型名称
        rewiring_flag: rewiring 标志 (True/False)
        just_reflection: 是否仅反思模式 (默认 False)
        anonymize_edges: 是否匿名化边名 (默认 False)
        webkb_full_abs: 是否显示全摘要 (默认 False)
    """
    suffix = "_rewired" if rewiring_flag else "_no_rewiring"

    # 按照 utils.py generate_pickle_filename 规则生成原始文件名
    original_filename = get_expected_pickle_filename(
        dataset, hop, iteration, mode, use_instructions, refining,
        model_full_name, just_reflection, anonymize_edges, webkb_full_abs
    )

    # 构造新文件名
    new_filename = original_filename.replace('.pkl', f'{suffix}.pkl')

    # 依次在当前目录和 result2/ 中查找
    search_paths = [original_filename, os.path.join('result2', original_filename)]
    for src_path in search_paths:
        if os.path.exists(src_path):
            dst_path = src_path.replace(original_filename, new_filename)
            try:
                os.rename(src_path, dst_path)
                print(f"  ✓ 重命名: {src_path} → {dst_path}")
                return True
            except Exception as e:
                print(f"  ✗ 重命名失败 ({src_path}): {e}")
                return False

    print(f"  ⚠ 文件不存在: {original_filename} (已查找: {search_paths})")
    return False


def run_single_experiment(model_name, model_full_name, dataset, rewiring,
                          hop, max_workers, iteration=1, use_instructions=True,
                          refining=True, mode='neighbors', just_reflection=False,
                          anonymize_edges=False, webkb_full_abs=False,
                          include_neighbors=False):
    """
    运行单个实验

    Args:
        model_name: 模型短名称 (用于显示)
        model_full_name: 模型完整名称 (用于API调用)
        dataset: 数据集名称
        rewiring: 是否启用rewiring
        hop: 跳数
        max_workers: 并发数
        iteration: 迭代次数
        use_instructions: 是否使用指导
        refining: 是否使用精炼
        mode: 模式 (ego/neighbors)
        just_reflection: 是否仅反思模式
        anonymize_edges: 是否匿名化边名
        webkb_full_abs: (WebKB) 是否显示全摘要
        include_neighbors: (WebKB) 是否包含邻居链接模式

    Returns:
        dict: 实验结果，包含状态、模型、数据集、耗时等信息
    """
    start_time = time.time()

    # 构建命令
    cmd = [
        sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_taptn.py'),
        '--dataset', dataset,
        '--model', model_full_name,
        '--hop', str(hop),
        '--max_workers', str(max_workers),
        '--iteration', str(iteration),
        '--mode', mode,
    ]

    # 添加可选参数
    if rewiring:
        cmd.append('--rewiring')

    if use_instructions:
        cmd.append('--use_instructions')

    if refining:
        cmd.append('--refining')

    if just_reflection:
        cmd.append('--just_reflection')

    if anonymize_edges:
        cmd.append('--anonymize_edges')

    if webkb_full_abs:
        cmd.append('--webkb_full_abs')

    if include_neighbors:
        cmd.append('--include_neighbors')

    # 其他默认参数
    cmd.extend([
        '--zero_shot_cot',
        '--include_abs',
    ])

    print(f"\n{'='*80}")
    print(f"开始实验: {model_name} on {dataset.upper()}")
    print(f"  Rewiring: {rewiring}")
    print(f"  Hop: {hop}")
    print(f"  Max Workers: {max_workers}")
    print(f"{'='*80}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    try:
        # 运行命令
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',  # WebKB 网页原文含非法 UTF-8 字节，需容错解码
            timeout=7200  # 2小时超时
        )

        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            print(f"\n✓ 实验完成: {model_name} on {dataset} ({elapsed_time:.1f}秒)")

            # 重命名pickle文件：rewired 运行由 generate_pickle_filename 自带 _rewired 后缀，
            # 仅 no_rewiring 运行需要补加 _no_rewiring 后缀（分析脚本按该后缀扫描）
            if not rewiring:
                rename_success = rename_pickle_files(
                    dataset, hop, iteration, mode, use_instructions,
                    refining, model_full_name, rewiring,
                    just_reflection, anonymize_edges, webkb_full_abs
                )
            else:
                rename_success = True
            return {
                'status': 'success',
                'model': model_name,
                'dataset': dataset,
                'rewiring': rewiring,
                'elapsed_time': elapsed_time,
                'renamed': rename_success
            }
        else:
            print(f"\n✗ 实验失败: {model_name} on {dataset}")
            print(f"错误输出:\n{result.stderr}")
            return {
                'status': 'failed',
                'model': model_name,
                'dataset': dataset,
                'rewiring': rewiring,
                'elapsed_time': elapsed_time,
                'error': result.stderr
            }
    
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        print(f"\n✗ 实验超时: {model_name} on {dataset} (>{elapsed_time:.1f}秒)")
        return {
            'status': 'timeout',
            'model': model_name,
            'dataset': dataset,
            'rewiring': rewiring,
            'elapsed_time': elapsed_time
        }
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ 实验异常: {model_name} on {dataset}")
        print(f"异常: {e}")
        return {
            'status': 'error',
            'model': model_name,
            'dataset': dataset,
            'rewiring': rewiring,
            'elapsed_time': elapsed_time,
            'error': str(e)
        }


def run_experiments_parallel(models, datasets, rewiring, hop, max_workers,
                            iteration, use_instructions, refining, mode,
                            batch_workers=20, just_reflection=False,
                            anonymize_edges=False, webkb_full_abs=False,
                            include_neighbors=False):
    """
    并发运行所有实验（两层并发：批量层 + 实验层）

    Args:
        models: 模型字典 {short_name: full_name}
        datasets: 数据集列表
        rewiring: 是否启用rewiring
        hop: 跳数
        max_workers: 每个实验的并发数（实验层）
        iteration: 迭代次数
        use_instructions: 是否使用指导
        refining: 是否使用精炼
        mode: 模式
        batch_workers: 批量层的并发数（多少个实验同时运行）
        just_reflection: 是否仅反思模式
        anonymize_edges: 是否匿名化边名
        webkb_full_abs: (WebKB) 是否显示全摘要
        include_neighbors: (WebKB) 是否包含邻居链接模式

    Returns:
        list: 所有实验结果
    """
    results = []
    total_experiments = len(models) * len(datasets)

    print(f"\n{'='*80}")
    print(f"批量实验配置")
    print(f"{'='*80}")
    print(f"模型数量: {len(models)}")
    print(f"数据集数量: {len(datasets)}")
    print(f"总实验数: {total_experiments}")
    print(f"Rewiring: {rewiring}")
    print(f"Hop: {hop}")
    print(f"批量层并发数: {batch_workers} (同时运行{batch_workers}个实验)")
    print(f"实验层并发数: {max_workers} (每个实验内{max_workers}线程)")
    print(f"{'='*80}\n")

    # 创建所有实验任务
    experiment_tasks = []
    experiment_idx = 0
    for model_name, model_full_name in models.items():
        for dataset in datasets:
            experiment_idx += 1
            experiment_tasks.append({
                'idx': experiment_idx,
                'model_name': model_name,
                'model_full_name': model_full_name,
                'dataset': dataset,
                'rewiring': rewiring,
                'hop': hop,
                'max_workers': max_workers,
                'iteration': iteration,
                'use_instructions': use_instructions,
                'refining': refining,
                'mode': mode,
                'just_reflection': just_reflection,
                'anonymize_edges': anonymize_edges,
                'webkb_full_abs': webkb_full_abs,
                'include_neighbors': include_neighbors,
            })

    # 使用线程池并发执行实验
    completed_count = 0
    failed_experiments = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_workers) as executor:
        # 提交所有任务
        future_to_task = {
            executor.submit(
                run_single_experiment,
                task['model_name'],
                task['model_full_name'],
                task['dataset'],
                task['rewiring'],
                task['hop'],
                task['max_workers'],
                task['iteration'],
                task['use_instructions'],
                task['refining'],
                task['mode'],
                task['just_reflection'],
                task['anonymize_edges'],
                task['webkb_full_abs'],
                task['include_neighbors'],
            ): task
            for task in experiment_tasks
        }
        
        # 收集结果
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            completed_count += 1
            
            try:
                result = future.result()
                results.append(result)
                
                status_symbol = "✓" if result['status'] == 'success' else "✗"
                print(f"\n[进度: {completed_count}/{total_experiments}] {status_symbol} "
                      f"{task['model_name']} on {task['dataset']} "
                      f"({result['status']}, {result['elapsed_time']:.1f}s)")
                
            except Exception as e:
                print(f"\n[进度: {completed_count}/{total_experiments}] ✗ "
                      f"{task['model_name']} on {task['dataset']} - 异常: {e}")
                failed_experiments.append({
                    'model': task['model_name'],
                    'dataset': task['dataset'],
                    'error': str(e)
                })
                results.append({
                    'status': 'error',
                    'model': task['model_name'],
                    'dataset': task['dataset'],
                    'rewiring': task['rewiring'],
                    'elapsed_time': 0,
                    'error': str(e)
                })
    
    if failed_experiments:
        print(f"\n{'='*80}")
        print(f"注意: {len(failed_experiments)} 个实验在并发执行时失败")
        print(f"{'='*80}")
    
    return results


def print_summary(results, start_time):
    """
    打印实验汇总
    
    Args:
        results: 实验结果列表
        start_time: 批量实验开始时间
    """
    total_time = time.time() - start_time
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = sum(1 for r in results if r['status'] == 'failed')
    timeout_count = sum(1 for r in results if r['status'] == 'timeout')
    error_count = sum(1 for r in results if r['status'] == 'error')
    
    renamed_count = sum(1 for r in results if r.get('renamed', False))
    
    print(f"\n\n{'='*80}")
    print(f"实验汇总")
    print(f"{'='*80}")
    print(f"总实验数: {len(results)}")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    print(f"超时: {timeout_count}")
    print(f"异常: {error_count}")
    print(f"文件重命名成功: {renamed_count}")
    print(f"总耗时: {total_time/60:.1f} 分钟")
    print(f"{'='*80}\n")
    
    # 详细结果
    if success_count > 0:
        print(f"✓ 成功的实验:")
        for r in results:
            if r['status'] == 'success':
                print(f"  - {r['model']} on {r['dataset']} "
                      f"(rewiring={r['rewiring']}, {r['elapsed_time']:.1f}s)")
    
    if failed_count > 0:
        print(f"\n✗ 失败的实验:")
        for r in results:
            if r['status'] == 'failed':
                print(f"  - {r['model']} on {r['dataset']} "
                      f"(rewiring={r['rewiring']}, {r['elapsed_time']:.1f}s)")
    
    if timeout_count > 0:
        print(f"\n⏱ 超时的实验:")
        for r in results:
            if r['status'] == 'timeout':
                print(f"  - {r['model']} on {r['dataset']} "
                      f"(rewiring={r['rewiring']}, {r['elapsed_time']:.1f}s)")
    
    if error_count > 0:
        print(f"\n⚠ 异常的实验:")
        for r in results:
            if r['status'] == 'error':
                print(f"  - {r['model']} on {r['dataset']} "
                      f"(rewiring={r['rewiring']}, {r['elapsed_time']:.1f}s)")
    
    print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description='批量运行 TAPTN 实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行 rewiring 实验
  %(prog)s --rewiring
  
  # 运行 no_rewiring 实验
  %(prog)s --no-rewiring
  
  # 自定义并发数
  %(prog)s --rewiring --max_workers 30
  
  # 仅运行特定模型
  %(prog)s --rewiring --models phi-4 gemma-2-9b-it
  
  # 仅运行特定数据集
  %(prog)s --rewiring --datasets cornell texas
        """
    )
    
    # Rewiring 参数（互斥）
    rewiring_group = parser.add_mutually_exclusive_group(required=False)
    rewiring_group.add_argument('--rewiring', action='store_true', default=False,
                                help='启用图重连 (生成 *_rewired.pkl)')
    rewiring_group.add_argument('--no-rewiring', action='store_true',
                                help='禁用图重连 (生成 *_no_rewiring.pkl)')
    
    # 实验参数
    parser.add_argument('--hop', type=int, default=1, choices=[1, 2],
                       help='邻居跳数 (默认: 2)')
    parser.add_argument('--max_workers', type=int, default=20,
                       help='每个实验内部的并发线程数 (默认: 20)')
    parser.add_argument('--batch_workers', type=int, default=20,
                       help='批量层并发数：同时运行多少个实验 (默认: 20)')
    parser.add_argument('--iteration', type=int, default=1, choices=[1, 2],
                       help='迭代次数 (默认: 1)')
    parser.add_argument('--mode', type=str, default='neighbors',
                       choices=['ego', 'neighbors'],
                       help='分类模式 (默认: neighbors)')
    
    # 可选参数
    parser.add_argument('--use-instructions', dest='use_instructions', action='store_true', default=False,
                       help='Enable step-by-step instructions (tb_3 / TAPTN). Default: off (tb_2 / label-free).')
    parser.add_argument('--no-instructions', dest='no_instructions', action='store_true', default=False,
                       help='Disable instructions (default).')
    parser.add_argument('--no-refining', action='store_true',
                       help='Disable refining (Section-2 rewiring keeps refining on).')
    parser.add_argument('--just_reflection', action='store_true', default=False,
                       help='仅反思模式，不使用邻居消息传递 (默认关闭)')
    parser.add_argument('--anonymize_edges', action='store_true', default=False,
                       help='匿名化边名称，用于消融实验 (默认关闭)')
    parser.add_argument('--webkb_full_abs', action='store_true', default=False,
                       help='(WebKB) label-free prompt: neighbor text + link stats, no neighbor category.')
    parser.add_argument('--include_neighbors', action='store_true', default=False,
                       help='(WebKB only) 在邻居描述中附加链接模式 (默认关闭)')

    # 过滤参数
    parser.add_argument('--models', nargs='+', choices=list(MODELS.keys()),
                       help='仅运行指定的模型 (默认: 全部)')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS,
                       help='仅运行指定的数据集 (默认: 全部)')

    args = parser.parse_args()
    # 确定rewiring标志
    rewiring_flag = args.rewiring

    # 确定要运行的模型和数据集
    models_to_run = {k: MODELS[k] for k in (args.models or MODELS.keys())}
    datasets_to_run = args.datasets or DATASETS

    # 其他参数
    use_instructions = bool(getattr(args, 'use_instructions', False)) and not args.no_instructions
    refining = not args.no_refining
    
    # 开始批量实验
    batch_start_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"批量 TAPTN 实验")
    print(f"{'='*80}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Rewiring: {rewiring_flag}")
    print(f"Hop: {args.hop}")
    print(f"批量层并发数: {args.batch_workers} (同时运行{args.batch_workers}个实验)")
    print(f"实验层并发数: {args.max_workers} (每个实验内{args.max_workers}线程)")
    print(f"模型: {list(models_to_run.keys())}")
    print(f"数据集: {datasets_to_run}")
    if args.just_reflection:
        print(f"模式: 仅反思 (just_reflection)")
    if args.anonymize_edges:
        print(f"匿名化边名: 开启")
    if args.webkb_full_abs:
        print(f"WebKB 全摘要: 开启")
    if args.include_neighbors:
        print(f"WebKB 链接模式: 开启")
    print(f"{'='*80}\n")

    # 并发运行所有实验（两层并发）
    results = run_experiments_parallel(
        models_to_run,
        datasets_to_run,
        rewiring_flag,
        args.hop,
        args.max_workers,
        args.iteration,
        use_instructions,
        refining,
        args.mode,
        args.batch_workers,
        just_reflection=args.just_reflection,
        anonymize_edges=args.anonymize_edges,
        webkb_full_abs=args.webkb_full_abs,
        include_neighbors=args.include_neighbors,
    )
    
    # 打印汇总
    print_summary(results, batch_start_time)
    
    print(f"\n批量实验完成!")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == '__main__':
    main()

