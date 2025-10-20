"""
工具函数
"""

import os
import yaml
import json
import torch
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件（支持 YAML 和 JSON）
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    if config_path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
    
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    保存配置文件
    
    Args:
        config: 配置字典
        save_path: 保存路径
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if save_path.suffix in ['.yaml', '.yml']:
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    elif save_path.suffix == '.json':
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    else:
        raise ValueError(f"不支持的配置文件格式: {save_path.suffix}")


def set_seed(seed: int):
    """
    设置随机种子以确保可复现性
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # 确保确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "auto") -> torch.device:
    """
    获取计算设备
    
    Args:
        device_str: 设备字符串 ("auto", "cuda", "cuda:0", "cpu")
        
    Returns:
        torch.device 对象
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    
    return device


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    统计模型参数量
    
    Args:
        model: PyTorch 模型
        
    Returns:
        参数统计字典
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': non_trainable,
        'total_mb': total * 4 / (1024 ** 2),  # 假设 float32
        'trainable_mb': trainable * 4 / (1024 ** 2)
    }


def format_time(seconds: float) -> str:
    """
    格式化时间
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.2f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}小时"


def get_lr(optimizer) -> float:
    """
    获取当前学习率
    
    Args:
        optimizer: 优化器
        
    Returns:
        当前学习率
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def print_model_structure(model: torch.nn.Module):
    """
    打印模型结构
    
    Args:
        model: PyTorch 模型
    """
    print("\n" + "=" * 80)
    print("模型结构")
    print("=" * 80)
    print(model)
    print("=" * 80)
    
    params = count_parameters(model)
    print(f"\n参数统计:")
    print(f"  总参数量:       {params['total']:>15,} ({params['total']/1e6:.2f}M)")
    print(f"  可训练参数:     {params['trainable']:>15,} ({params['trainable']/1e6:.2f}M)")
    print(f"  不可训练参数:   {params['non_trainable']:>15,}")
    print(f"  模型大小:       {params['total_mb']:>15.2f} MB")
    print("=" * 80 + "\n")

