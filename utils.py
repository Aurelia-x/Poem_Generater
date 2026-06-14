import json
import os
import random

import numpy as np
import torch


# ==========================================
#  通用工具函数
# ==========================================

def ensure_dir(path):
    """
    确保目录存在；若不存在则自动创建

    参数:
        path: 目录路径
    """
    os.makedirs(path, exist_ok=True)


def set_seed(seed):
    """
    设置随机种子，保证实验可复现

    参数:
        seed: 随机种子值
    """
    print(f"\n[utils.set_seed] 开始设置随机种子...")
    print(f"  [utils.set_seed] seed = {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"[utils.set_seed] 随机种子设置完成")


def get_device():
    """
    获取当前训练设备

    返回:
        device: "cuda" 或 "cpu"
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def save_json(obj, file_path):
    """
    保存 JSON 文件

    参数:
        obj:       待保存对象
        file_path: 保存路径
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(file_path):
    """
    读取 JSON 文件

    参数:
        file_path: 文件路径

    返回:
        反序列化后的对象
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_parameters(model):
    """
    统计模型参数量

    参数:
        model: 模型实例

    返回:
        total_params:     总参数量
        trainable_params: 可训练参数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
