import json
import os
import random
import sys

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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch_runtime():
    """
    配置 PyTorch 运行时

    说明:
        在当前 Windows + PyTorch 2.11 + CUDA 环境下，
        较大尺寸的 cuDNN LSTM 在进程退出时会触发原生崩溃，
        表现为终端退出码 -1073740791。

        为保证训练 / 评估 / 生成脚本稳定退出，这里统一禁用
        cuDNN 的 RNN 路径，改用更稳定的 CUDA 实现。
    """
    if torch.cuda.is_available() and sys.platform.startswith("win"):
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False


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
