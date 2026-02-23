#!/usr/bin/env python3

"""
数据集信息查看脚本

查看 LIBERO 各个 Suite 的详细信息，包括任务数量、数据集大小等。
"""

import sys


def print_libero_info():
    """打印 LIBERO 数据集信息"""
    
    print("=" * 80)
    print("LIBERO 数据集信息")
    print("=" * 80)
    print()
    
    suites = {
        "libero_spatial": {
            "description": "空间关系任务（Spatial Reasoning）",
            "tasks": 10,
            "examples": [
                "把红色方块放在蓝色方块的左边",
                "把物体放在容器的前面",
            ],
            "dataset_id": "lerobot/libero_spatial_no_noops",
            "max_steps": 280,
        },
        "libero_object": {
            "description": "物体操作任务（Object Manipulation）",
            "tasks": 10,
            "examples": [
                "拿起红色方块",
                "把物体放进容器",
            ],
            "dataset_id": "lerobot/libero_object_no_noops",
            "max_steps": 280,
        },
        "libero_goal": {
            "description": "目标导向任务（Goal-Oriented）",
            "tasks": 10,
            "examples": [
                "把所有物体放进容器",
                "清理桌面",
            ],
            "dataset_id": "lerobot/libero_goal_no_noops",
            "max_steps": 300,
        },
        "libero_10": {
            "description": "10 个混合任务（10-Task Mix）",
            "tasks": 10,
            "examples": [
                "复杂的多步骤操作",
                "组合多个子任务",
            ],
            "dataset_id": "lerobot/libero_10_no_noops",
            "max_steps": 520,
        },
        "libero_90": {
            "description": "90 个混合任务（90-Task Mix）",
            "tasks": 90,
            "examples": [
                "大规模任务集合",
                "涵盖各种操作类型",
            ],
            "dataset_id": "lerobot/libero_90_no_noops",
            "max_steps": 400,
        },
    }
    
    for suite_name, info in suites.items():
        print(f"【{suite_name}】")
        print(f"  描述: {info['description']}")
        print(f"  任务数量: {info['tasks']}")
        print(f"  最大步数: {info['max_steps']}")
        print(f"  数据集 ID: {info['dataset_id']}")
        print(f"  示例任务:")
        for example in info['examples']:
            print(f"    - {example}")
        print()
    
    print("=" * 80)
    print("实验设计说明")
    print("=" * 80)
    print()
    print("本实验使用以下设置:")
    print()
    print("  ID (In-Domain) 训练集:")
    print("    - libero_spatial")
    print("    - 用于微调 Baseline 和 CRaFT")
    print()
    print("  OOD (Out-of-Distribution) 测试集:")
    print("    - libero_object")
    print("    - libero_goal")
    print("    - libero_10")
    print("    - 用于评估跨 Suite 泛化能力")
    print()
    print("  实验目标:")
    print("    - 验证 CRaFT 在 ID 任务上保持性能")
    print("    - 验证 CRaFT 在 OOD 任务上显著优于 Baseline")
    print()
    print("=" * 80)
    print()


def check_dataset_availability():
    """检查数据集是否可用"""
    
    print("检查数据集可用性...")
    print()
    
    try:
        from huggingface_hub import list_repo_files
        
        datasets = [
            "lerobot/libero_spatial_no_noops",
            "lerobot/libero_object_no_noops",
            "lerobot/libero_goal_no_noops",
            "lerobot/libero_10_no_noops",
        ]
        
        for dataset_id in datasets:
            try:
                files = list_repo_files(dataset_id, repo_type="dataset")
                print(f"  ✓ {dataset_id} (可用)")
            except Exception as e:
                print(f"  ✗ {dataset_id} (不可用: {e})")
        
        print()
        
    except ImportError:
        print("  提示: 安装 huggingface_hub 以检查数据集可用性")
        print("  pip install huggingface_hub")
        print()


def main():
    """主函数"""
    
    print_libero_info()
    check_dataset_availability()
    
    print("提示:")
    print("  - 首次运行训练时，数据集会自动从 HuggingFace Hub 下载")
    print("  - 下载速度取决于网络连接")
    print("  - 数据集会缓存在本地，后续运行无需重新下载")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

