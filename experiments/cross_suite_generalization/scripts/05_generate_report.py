#!/usr/bin/env python3

"""
步骤 5: 生成对比报告

从评测结果中提取成功率数据，生成对比表格和可视化图表。
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_eval_result(result_path: Path) -> Dict:
    """加载评测结果 JSON 文件"""
    if not result_path.exists():
        print(f"警告: 结果文件不存在: {result_path}")
        return None
    
    with open(result_path, 'r') as f:
        return json.load(f)


def extract_success_rate(eval_info: Dict) -> float:
    """从评测结果中提取成功率"""
    if eval_info is None:
        return 0.0
    
    # 尝试多种可能的字段名
    for key in ['avg_success_rate', 'success_rate', 'avg_success', 'success']:
        if key in eval_info:
            return float(eval_info[key]) * 100  # 转换为百分比
    
    # 如果没有找到，尝试从 per_episode 数据计算
    if 'per_episode' in eval_info:
        episodes = eval_info['per_episode']
        if isinstance(episodes, list) and len(episodes) > 0:
            successes = sum(1 for ep in episodes if ep.get('success', False))
            return (successes / len(episodes)) * 100
    
    print(f"警告: 无法从结果中提取成功率")
    return 0.0


def collect_results(results_dir: Path) -> Tuple[Dict[str, float], Dict[str, float]]:
    """收集所有评测结果"""
    
    suites = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
    
    baseline_results = {}
    craft_results = {}
    
    for suite in suites:
        # Baseline 结果
        baseline_path = results_dir / f"baseline_spatial_on_{suite}" / "eval_info.json"
        baseline_info = load_eval_result(baseline_path)
        baseline_results[suite] = extract_success_rate(baseline_info)
        
        # CRaFT 结果
        craft_path = results_dir / f"craft_spatial_on_{suite}" / "eval_info.json"
        craft_info = load_eval_result(craft_path)
        craft_results[suite] = extract_success_rate(craft_info)
    
    return baseline_results, craft_results


def generate_markdown_report(
    baseline_results: Dict[str, float],
    craft_results: Dict[str, float],
    output_path: Path
):
    """生成 Markdown 格式的对比报告"""
    
    report = []
    report.append("# CRaFT 跨 Suite 泛化能力验证 - 实验报告")
    report.append("")
    report.append("---")
    report.append("")
    
    # 实验设置
    report.append("## 实验设置")
    report.append("")
    report.append("| 项目 | 配置 |")
    report.append("|------|------|")
    report.append("| **Base Model** | `lerobot/pi0_fast` |")
    report.append("| **ID 训练集** | `libero_spatial` |")
    report.append("| **OOD 测试集** | `libero_object`, `libero_goal`, `libero_10` |")
    report.append("| **训练步数** | 10,000 |")
    report.append("| **评测 Episodes** | 50 per suite |")
    report.append("")
    
    # 结果对比表格
    report.append("## 结果对比")
    report.append("")
    report.append("| Suite | Type | Baseline Success Rate | CRaFT Success Rate | Improvement |")
    report.append("|-------|------|----------------------|-------------------|-------------|")
    
    suites_info = [
        ('libero_spatial', 'ID'),
        ('libero_object', 'OOD'),
        ('libero_goal', 'OOD'),
        ('libero_10', 'OOD'),
    ]
    
    total_improvement = 0.0
    ood_count = 0
    
    for suite, suite_type in suites_info:
        baseline_sr = baseline_results.get(suite, 0.0)
        craft_sr = craft_results.get(suite, 0.0)
        improvement = craft_sr - baseline_sr
        
        if suite_type == 'OOD':
            total_improvement += improvement
            ood_count += 1
        
        # 格式化改进值（正数用绿色标记）
        if improvement > 0:
            improvement_str = f"**+{improvement:.1f}%**"
        elif improvement < 0:
            improvement_str = f"{improvement:.1f}%"
        else:
            improvement_str = "0.0%"
        
        report.append(
            f"| {suite} | {suite_type} | {baseline_sr:.1f}% | {craft_sr:.1f}% | {improvement_str} |"
        )
    
    report.append("")
    
    # 平均改进
    avg_ood_improvement = total_improvement / ood_count if ood_count > 0 else 0.0
    report.append("### 关键指标")
    report.append("")
    report.append(f"- **OOD 平均改进**: +{avg_ood_improvement:.1f}%")
    report.append("")
    
    # 结论
    report.append("## 结论")
    report.append("")
    
    id_baseline = baseline_results.get('libero_spatial', 0.0)
    id_craft = craft_results.get('libero_spatial', 0.0)
    id_diff = abs(id_craft - id_baseline)
    
    if id_diff < 5.0:
        report.append("✅ **ID 性能保持**: CRaFT 在 `libero_spatial` 上的性能与 Baseline 相当")
    else:
        report.append(f"⚠️ **ID 性能变化**: CRaFT 在 `libero_spatial` 上的性能与 Baseline 相差 {id_diff:.1f}%")
    
    report.append("")
    
    if avg_ood_improvement > 5.0:
        report.append(f"✅ **OOD 泛化提升**: CRaFT 在所有 OOD Suites 上的平均成功率提升 {avg_ood_improvement:.1f}%")
        report.append("")
        report.append("✅ **验证论文主张**: CRaFT 有效缓解灾难性遗忘，显著提升跨 Suite 泛化能力")
    else:
        report.append(f"⚠️ **OOD 泛化有限**: CRaFT 在 OOD Suites 上的平均改进仅为 {avg_ood_improvement:.1f}%")
        report.append("")
        report.append("⚠️ **需要调优**: 建议调整 CRaFT 超参数或增加训练步数")
    
    report.append("")
    
    # 详细分析
    report.append("## 详细分析")
    report.append("")
    
    for suite, suite_type in suites_info:
        baseline_sr = baseline_results.get(suite, 0.0)
        craft_sr = craft_results.get(suite, 0.0)
        improvement = craft_sr - baseline_sr
        
        report.append(f"### {suite} ({suite_type})")
        report.append("")
        report.append(f"- Baseline: {baseline_sr:.1f}%")
        report.append(f"- CRaFT: {craft_sr:.1f}%")
        report.append(f"- Improvement: {improvement:+.1f}%")
        report.append("")
        
        if suite_type == 'ID':
            if id_diff < 5.0:
                report.append("**分析**: ID 性能保持良好，CRaFT 没有在目标任务上造成性能下降。")
            else:
                report.append("**分析**: ID 性能有明显变化，可能需要调整 CRaFT 超参数。")
        else:
            if improvement > 10.0:
                report.append("**分析**: OOD 泛化能力显著提升，CRaFT 有效保留了预训练知识。")
            elif improvement > 0:
                report.append("**分析**: OOD 泛化能力有所提升，但改进幅度有限。")
            else:
                report.append("**分析**: OOD 泛化能力未见提升，可能需要调整训练策略。")
        
        report.append("")
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"✓ Markdown 报告已生成: {output_path}")


def generate_csv_table(
    baseline_results: Dict[str, float],
    craft_results: Dict[str, float],
    output_path: Path
):
    """生成 CSV 格式的对比表格"""
    
    import csv
    
    suites_info = [
        ('libero_spatial', 'ID'),
        ('libero_object', 'OOD'),
        ('libero_goal', 'OOD'),
        ('libero_10', 'OOD'),
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Suite', 'Type', 'Baseline Success Rate (%)', 'CRaFT Success Rate (%)', 'Improvement (%)'])
        
        for suite, suite_type in suites_info:
            baseline_sr = baseline_results.get(suite, 0.0)
            craft_sr = craft_results.get(suite, 0.0)
            improvement = craft_sr - baseline_sr
            
            writer.writerow([suite, suite_type, f"{baseline_sr:.1f}", f"{craft_sr:.1f}", f"{improvement:+.1f}"])
    
    print(f"✓ CSV 表格已生成: {output_path}")


def generate_visualization(
    baseline_results: Dict[str, float],
    craft_results: Dict[str, float],
    output_path: Path
):
    """生成可视化对比图"""
    
    suites = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
    suite_labels = ['Spatial\n(ID)', 'Object\n(OOD)', 'Goal\n(OOD)', '10-Task\n(OOD)']
    
    baseline_values = [baseline_results.get(s, 0.0) for s in suites]
    craft_values = [craft_results.get(s, 0.0) for s in suites]
    
    x = np.arange(len(suites))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline (Naive SFT)', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x + width/2, craft_values, width, label='CRaFT', color='#4ECDC4', alpha=0.8)
    
    ax.set_xlabel('Suite', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('CRaFT vs Baseline: Cross-Suite Generalization', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(suite_labels)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 可视化图表已生成: {output_path}")


def main():
    """主函数"""
    
    print("=" * 60)
    print("生成对比报告")
    print("=" * 60)
    print()
    
    # 设置路径
    experiment_dir = Path("experiments/cross_suite_generalization")
    results_dir = experiment_dir / "results"
    
    if not results_dir.exists():
        print(f"错误: 结果目录不存在: {results_dir}")
        sys.exit(1)
    
    # 收集结果
    print("收集评测结果...")
    baseline_results, craft_results = collect_results(results_dir)
    
    print()
    print("Baseline 结果:")
    for suite, sr in baseline_results.items():
        print(f"  {suite}: {sr:.1f}%")
    
    print()
    print("CRaFT 结果:")
    for suite, sr in craft_results.items():
        print(f"  {suite}: {sr:.1f}%")
    
    print()
    
    # 生成报告
    print("生成报告文件...")
    
    # Markdown 报告
    markdown_path = results_dir / "comparison_report.md"
    generate_markdown_report(baseline_results, craft_results, markdown_path)
    
    # CSV 表格
    csv_path = results_dir / "comparison_table.csv"
    generate_csv_table(baseline_results, craft_results, csv_path)
    
    # 可视化图表
    try:
        viz_path = results_dir / "success_rate_comparison.png"
        generate_visualization(baseline_results, craft_results, viz_path)
    except Exception as e:
        print(f"警告: 生成可视化图表失败: {e}")
        print("提示: 请确保已安装 matplotlib: pip install matplotlib")
    
    print()
    print("=" * 60)
    print("报告生成完成！")
    print("=" * 60)
    print()
    print("输出文件:")
    print(f"  - {markdown_path}")
    print(f"  - {csv_path}")
    if (results_dir / "success_rate_comparison.png").exists():
        print(f"  - {results_dir / 'success_rate_comparison.png'}")
    print()
    
    # 打印快速摘要
    print("快速摘要:")
    print("-" * 60)
    
    id_suite = 'libero_spatial'
    id_baseline = baseline_results.get(id_suite, 0.0)
    id_craft = craft_results.get(id_suite, 0.0)
    id_diff = id_craft - id_baseline
    
    print(f"ID 性能 ({id_suite}):")
    print(f"  Baseline: {id_baseline:.1f}%  |  CRaFT: {id_craft:.1f}%  |  Diff: {id_diff:+.1f}%")
    print()
    
    ood_suites = ['libero_object', 'libero_goal', 'libero_10']
    print("OOD 泛化:")
    
    total_improvement = 0.0
    for suite in ood_suites:
        baseline_sr = baseline_results.get(suite, 0.0)
        craft_sr = craft_results.get(suite, 0.0)
        improvement = craft_sr - baseline_sr
        total_improvement += improvement
        
        print(f"  {suite}:")
        print(f"    Baseline: {baseline_sr:.1f}%  |  CRaFT: {craft_sr:.1f}%  |  Improvement: {improvement:+.1f}%")
    
    avg_improvement = total_improvement / len(ood_suites)
    print()
    print(f"OOD 平均改进: {avg_improvement:+.1f}%")
    print("-" * 60)
    print()


if __name__ == "__main__":
    main()

