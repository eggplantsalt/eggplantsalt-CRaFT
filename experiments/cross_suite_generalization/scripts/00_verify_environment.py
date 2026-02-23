#!/usr/bin/env python3

"""
快速验证脚本

在运行完整实验前，快速验证环境配置是否正确。
"""

import sys
from pathlib import Path


def check_python_version():
    """检查 Python 版本"""
    print("检查 Python 版本...", end=" ")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (需要 >= 3.8)")
        return False


def check_module(module_name, package_name=None):
    """检查 Python 模块是否安装"""
    if package_name is None:
        package_name = module_name
    
    print(f"检查 {module_name}...", end=" ")
    try:
        __import__(module_name)
        print("✓")
        return True
    except ImportError:
        print(f"✗ (请运行: pip install {package_name})")
        return False


def check_cuda():
    """检查 CUDA 是否可用"""
    print("检查 CUDA...", end=" ")
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"✓ {device_name}")
            return True
        else:
            print("✗ CUDA 不可用（将使用 CPU，速度会很慢）")
            return False
    except Exception as e:
        print(f"✗ 错误: {e}")
        return False


def check_directory_structure():
    """检查目录结构"""
    print("检查目录结构...", end=" ")
    
    experiment_dir = Path("experiments/cross_suite_generalization")
    
    required_dirs = [
        experiment_dir,
        experiment_dir / "scripts",
        experiment_dir / "configs",
    ]
    
    required_files = [
        experiment_dir / "README.md",
        experiment_dir / "run_full_experiment.sh",
        experiment_dir / "scripts/01_train_baseline.sh",
        experiment_dir / "scripts/02_build_anchor_cache.sh",
        experiment_dir / "scripts/03_train_craft.sh",
        experiment_dir / "scripts/04_eval_cross_suite.sh",
        experiment_dir / "scripts/05_generate_report.py",
        experiment_dir / "configs/baseline_spatial.yaml",
        experiment_dir / "configs/craft_spatial.yaml",
    ]
    
    all_exist = True
    
    for d in required_dirs:
        if not d.exists():
            print(f"\n  ✗ 缺失目录: {d}")
            all_exist = False
    
    for f in required_files:
        if not f.exists():
            print(f"\n  ✗ 缺失文件: {f}")
            all_exist = False
    
    if all_exist:
        print("✓")
    
    return all_exist


def check_lerobot_scripts():
    """检查 LeRobot 脚本是否存在"""
    print("检查 LeRobot 脚本...", end=" ")
    
    required_scripts = [
        "src/lerobot/scripts/lerobot_train.py",
        "src/lerobot/scripts/lerobot_train_craft.py",
        "src/lerobot/scripts/lerobot_eval.py",
        "src/lerobot/scripts/build_anchor_hidden_cache.py",
    ]
    
    all_exist = True
    
    for script in required_scripts:
        if not Path(script).exists():
            print(f"\n  ✗ 缺失脚本: {script}")
            all_exist = False
    
    if all_exist:
        print("✓")
    
    return all_exist


def main():
    """主函数"""
    
    print("=" * 60)
    print("CRaFT 跨 Suite 泛化实验 - 环境验证")
    print("=" * 60)
    print()
    
    checks = []
    
    # 基础检查
    checks.append(("Python 版本", check_python_version()))
    checks.append(("PyTorch", check_module("torch")))
    checks.append(("CUDA", check_cuda()))
    checks.append(("LeRobot", check_module("lerobot")))
    checks.append(("LIBERO", check_module("libero", "libero-robotics")))
    checks.append(("NumPy", check_module("numpy")))
    checks.append(("Matplotlib", check_module("matplotlib")))
    
    # 目录和文件检查
    checks.append(("目录结构", check_directory_structure()))
    checks.append(("LeRobot 脚本", check_lerobot_scripts()))
    
    print()
    print("=" * 60)
    
    # 统计结果
    passed = sum(1 for _, result in checks if result)
    total = len(checks)
    
    if passed == total:
        print(f"✓ 所有检查通过 ({passed}/{total})")
        print()
        print("环境配置正确，可以开始实验！")
        print()
        print("运行完整实验:")
        print("  bash experiments/cross_suite_generalization/run_full_experiment.sh")
        print()
        print("或分步运行:")
        print("  bash experiments/cross_suite_generalization/scripts/01_train_baseline.sh")
        print("  bash experiments/cross_suite_generalization/scripts/02_build_anchor_cache.sh")
        print("  bash experiments/cross_suite_generalization/scripts/03_train_craft.sh")
        print("  bash experiments/cross_suite_generalization/scripts/04_eval_cross_suite.sh")
        print("  python experiments/cross_suite_generalization/scripts/05_generate_report.py")
        print()
        return 0
    else:
        print(f"✗ 部分检查失败 ({passed}/{total})")
        print()
        print("请根据上述提示修复问题后重试。")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())

