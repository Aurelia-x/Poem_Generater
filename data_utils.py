import os
import re
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import (
    BATCH_SIZE,
    BOS_TOKEN,
    CLEAN_DATA_PATH,
    EOS_TOKEN,
    MAX_POEMS,
    MIN_CHAR_FREQ,
    PAD_TOKEN,
    POEM_CHAR_LEN,
    RAW_DATA_PATH,
    SPECIAL_TOKENS,
    TEST_RATIO,
    TRAIN_RATIO,
    UNK_TOKEN,
    VAL_RATIO,
    VOCAB_PATH,
)
from utils import ensure_dir, save_json


# ==========================================
#  1. 读取与清洗诗歌数据
# ==========================================

def clean_poem_line(line):
    """
    清洗单行诗歌文本

    清洗规则:
        1. 去掉首尾空白
        2. 仅保留中文汉字，过滤标点、空格、数字等

    参数:
        line: 原始文本行

    返回:
        cleaned: 清洗后的纯汉字字符串
    """
    line = line.strip()
    cleaned = re.sub(r"[^\u4e00-\u9fff]", "", line)
    return cleaned


def is_valid_qiyan_jueju(text):
    """
    判断一首诗是否满足七言绝句长度要求

    参数:
        text: 清洗后的纯汉字文本

    返回:
        bool: 是否恰好 28 字
    """
    return len(text) == POEM_CHAR_LEN


def load_and_clean_poems(file_path=RAW_DATA_PATH, max_poems=MAX_POEMS):
    """
    读取原始诗歌数据并清洗，只保留七言绝句

    参数:
        file_path: 原始数据路径
        max_poems: 最多保留多少首诗，None 表示全量

    返回:
        valid_poems: 过滤后的七言绝句列表
    """
    print("\n" + "=" * 55)
    print(" [数据读取与清洗]")
    print("=" * 55)

    print(f"\n[数据读取] 原始文件路径: {file_path}")
    assert os.path.exists(file_path), f"[数据读取] 文件不存在: {file_path}"

    with open(file_path, "r", encoding="utf-8") as f:
        raw_lines = [line.rstrip("\n") for line in f]

    print(f"[数据读取] 原始文本行数: {len(raw_lines)}")

    cleaned_poems = []
    invalid_count = 0

    for line in raw_lines:
        cleaned = clean_poem_line(line)
        if is_valid_qiyan_jueju(cleaned):
            cleaned_poems.append(cleaned)
        else:
            invalid_count += 1

    if max_poems is not None:
        cleaned_poems = cleaned_poems[:max_poems]

    print(f"[数据清洗] 满足七言绝句长度要求的样本数: {len(cleaned_poems)}")
    print(f"[数据清洗] 被过滤的无效样本数: {invalid_count}")
    print(f"[数据清洗] 最大使用样本数限制: {max_poems}")

    assert len(cleaned_poems) > 0, "[数据清洗] 过滤后没有可用样本!"

    print(f"\n[数据清洗] 前 3 首有效样本预览:")
    for idx, poem in enumerate(cleaned_poems[:3]):
        print(f"  样本 {idx}: {poem}  (长度={len(poem)})")

    return cleaned_poems


# ==========================================
#  2. 构建词表与编码
# ==========================================

def build_vocab(poems, min_freq=MIN_CHAR_FREQ):
    """
    基于训练语料构建字符表

    参数:
        poems:    诗歌文本列表
        min_freq: 最小词频阈值

    返回:
        stoi: 字符 -> 索引
        itos: 索引 -> 字符
    """
    print("\n" + "=" * 55)
    print(" [构建字符表]")
    print("=" * 55)

    counter = Counter()
    for poem in poems:
        counter.update(list(poem))

    vocab_chars = [char for char, freq in counter.items() if freq >= min_freq]
    vocab_chars = sorted(vocab_chars)

    itos = SPECIAL_TOKENS + vocab_chars
    stoi = {char: idx for idx, char in enumerate(itos)}

    print(f"\n[词表] 特殊 token 数量: {len(SPECIAL_TOKENS)}")
    print(f"[词表] 普通汉字数量:     {len(vocab_chars)}")
    print(f"[词表] 总词表大小 V =   {len(itos)}")
    print(f"[词表] 词频最高的前 10 个字:")
    for char, freq in counter.most_common(10):
        print(f"  '{char}' -> {freq}")

    print(f"\n[词表] 前 15 个 token 映射预览:")
    for idx, token in enumerate(itos[:15]):
        print(f"  {idx:>3d} -> {token}")

    return stoi, itos


def encode_poem(poem, stoi):
    """
    将一首诗编码为整数序列

    编码格式:
        [BOS] + 28 个汉字 + [EOS]

    参数:
        poem: 清洗后的 28 字诗歌
        stoi: 字符到索引的映射

    返回:
        encoded: 长度 30 的整数序列
    """
    unk_id = stoi[UNK_TOKEN]
    bos_id = stoi[BOS_TOKEN]
    eos_id = stoi[EOS_TOKEN]

    encoded = [bos_id]
    encoded.extend(stoi.get(char, unk_id) for char in poem)
    encoded.append(eos_id)

    assert len(encoded) == POEM_CHAR_LEN + 2, \
        f"[encode_poem] 编码长度异常: 期望 {POEM_CHAR_LEN + 2}, 实际 {len(encoded)}"

    return encoded


def decode_ids(ids, itos):
    """
    将整数序列解码回字符串（忽略特殊 token）

    参数:
        ids:  整数序列
        itos: 索引到字符的映射

    返回:
        text: 解码后的文本
    """
    ignore_tokens = {PAD_TOKEN, BOS_TOKEN, EOS_TOKEN}
    chars = []
    for idx in ids:
        token = itos[idx]
        if token not in ignore_tokens:
            chars.append(token)
    return "".join(chars)


# ==========================================
#  3. 数据集划分与 Dataset
# ==========================================

def split_dataset(poems, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO):
    """
    划分训练 / 验证 / 测试集

    参数:
        poems:       诗歌样本列表
        train_ratio: 训练集比例
        val_ratio:   验证集比例
        test_ratio:  测试集比例

    返回:
        train_poems, val_poems, test_poems
    """
    print("\n" + "=" * 55)
    print(" [划分数据集]")
    print("=" * 55)

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8, \
        "[数据划分] train/val/test 比例之和必须为 1"

    poems = list(poems)
    np.random.shuffle(poems)

    total = len(poems)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_poems = poems[:train_end]
    val_poems = poems[train_end:val_end]
    test_poems = poems[val_end:]

    print(f"\n[数据划分] 总样本数: {total}")
    print(f"[数据划分] 训练集:   {len(train_poems)}")
    print(f"[数据划分] 验证集:   {len(val_poems)}")
    print(f"[数据划分] 测试集:   {len(test_poems)}")

    assert len(train_poems) > 0, "[数据划分] 训练集为空!"
    assert len(val_poems) > 0, "[数据划分] 验证集为空!"
    assert len(test_poems) > 0, "[数据划分] 测试集为空!"

    return train_poems, val_poems, test_poems


class PoemDataset(Dataset):
    """
    诗歌语言模型数据集

    对每首长度 30 的编码序列:
        seq = [BOS] + 28字 + [EOS]

    构造:
        x = seq[:-1]  -> 长度 29
        y = seq[1:]   -> 长度 29
    """

    def __init__(self, poems, stoi):
        super(PoemDataset, self).__init__()
        self.poems = poems
        self.stoi = stoi
        self._shape_checked = False

        self.encoded_poems = [encode_poem(poem, stoi) for poem in poems]

        print(f"\n[PoemDataset __init__] 数据集构建完成")
        print(f"  [PoemDataset] 样本数: {len(self.encoded_poems)}")

    def __len__(self):
        return len(self.encoded_poems)

    def __getitem__(self, idx):
        seq = self.encoded_poems[idx]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)

        if not self._shape_checked:
            print(f"  [PoemDataset __getitem__] shape 断言通过: x {tuple(x.shape)}, y {tuple(y.shape)}")
            self._shape_checked = True

        return x, y


def build_dataloaders():
    """
    构建训练 / 验证 / 测试 DataLoader，并缓存清洗结果与词表

    返回:
        train_loader, val_loader, test_loader, stoi, itos
    """
    cache_dir = os.path.dirname(CLEAN_DATA_PATH)
    ensure_dir(cache_dir)

    poems = load_and_clean_poems()
    train_poems, val_poems, test_poems = split_dataset(poems)

    stoi, itos = build_vocab(train_poems)

    with open(CLEAN_DATA_PATH, "w", encoding="utf-8") as f:
        for poem in poems:
            f.write(poem + "\n")
    save_json({"stoi": stoi, "itos": itos}, VOCAB_PATH)

    print(f"\n[缓存] 清洗后诗歌已保存到: {CLEAN_DATA_PATH}")
    print(f"[缓存] 词表映射已保存到:     {VOCAB_PATH}")

    train_dataset = PoemDataset(train_poems, stoi)
    val_dataset = PoemDataset(val_poems, stoi)
    test_dataset = PoemDataset(test_poems, stoi)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("\n" + "=" * 55)
    print(" [DataLoader 构建完成]")
    print("=" * 55)
    print(f"\n[DataLoader] Batch Size: {BATCH_SIZE}")
    print(f"[DataLoader] train_loader batch 数: {len(train_loader)}")
    print(f"[DataLoader] val_loader   batch 数: {len(val_loader)}")
    print(f"[DataLoader] test_loader  batch 数: {len(test_loader)}")

    sample_x, sample_y = train_dataset[0]
    print(f"\n[DataLoader] 样本维度预览:")
    print(f"  x shape: {tuple(sample_x.shape)}  (期望: (29,))")
    print(f"  y shape: {tuple(sample_y.shape)}  (期望: (29,))")

    return train_loader, val_loader, test_loader, stoi, itos
