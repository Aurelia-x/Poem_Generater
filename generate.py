import argparse
import os

import torch

from config import (
    BASE_DIR,
    BEST_MODEL_SAVE_PATH,
    BOS_TOKEN,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    EOS_TOKEN,
    OUTPUT_DIR,
    PAD_TOKEN,
    SAMPLES_DIR,
    UNK_TOKEN,
)
from data_utils import decode_ids
from metrics import analyze_generated_poem
from model import build_model
from sampler import sample_next_token
from utils import configure_torch_runtime, ensure_dir, get_device, set_seed


# ==========================================
#  1. 加载模型与词表
# ==========================================

def load_model_from_checkpoint(checkpoint_path, device):
    """
    从 checkpoint 加载模型与词表

    参数:
        checkpoint_path: checkpoint 路径
        device:          计算设备

    返回:
        model:      已加载参数的模型
        checkpoint: checkpoint 字典
        stoi:       字符 -> 索引映射
        itos:       索引 -> 字符映射
    """
    print("\n" + "=" * 55)
    print(" [加载 checkpoint]")
    print("=" * 55)

    print(f"\n[模型加载] checkpoint 路径: {checkpoint_path}")
    assert os.path.exists(checkpoint_path), f"[模型加载] checkpoint 不存在: {checkpoint_path}"

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]

    model = build_model(
        vocab_size=config["vocab_size"],
        model_type=config["model_type"],
        embed_dim=config["embed_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"[模型加载] 模型参数加载成功")
    print(f"[模型加载] 训练 epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"[模型加载] best val ppl: {checkpoint.get('best_val_ppl', 'N/A')}")
    print(f"[模型加载] vocab size: {len(itos)}")

    return model, checkpoint, stoi, itos


# ==========================================
#  2. 基础逐字生成函数
# ==========================================

@torch.no_grad()
def generate_sequence(model, prompt_ids, stoi, itos, device,
                      max_new_tokens=28, strategy="temperature",
                      temperature=0.8, top_k=10, stop_at_eos=True):
    """
    从 prompt 开始，逐字生成后续 token

    参数:
        model:           诗歌语言模型
        prompt_ids:      初始 prompt 的 token id 列表
        stoi, itos:      词表映射
        device:          计算设备
        max_new_tokens:  最多新生成多少个 token
        strategy:        采样策略
        temperature:     温度系数
        top_k:           top-k 参数
        stop_at_eos:     是否在遇到 EOS 时提前停止

    返回:
        generated_ids: 完整 token 序列（包含 prompt）
    """
    generated_ids = list(prompt_ids)

    for step in range(max_new_tokens):
        input_tensor = torch.tensor([generated_ids], dtype=torch.long, device=device)
        logits, _ = model(input_tensor)
        next_logits = logits[0, -1, :]

        next_id = sample_next_token(
            next_logits,
            strategy=strategy,
            temperature=temperature,
            top_k=top_k,
        )

        generated_ids.append(next_id)

        if stop_at_eos and next_id == stoi[EOS_TOKEN]:
            break

    return generated_ids


def extract_text_from_ids(generated_ids, itos):
    """
    从 token id 序列中提取文本，忽略特殊 token
    """
    return decode_ids(generated_ids, itos)


def format_poem_for_display(text):
    """
    将 28 字诗歌按每句 7 字切分，并在句间加入一个空格

    参数:
        text: 原始连续诗歌字符串

    返回:
        formatted: 带空格的展示文本
    """
    if len(text) <= 7:
        return text

    lines = [text[i:i + 7] for i in range(0, len(text), 7)]
    return " ".join(lines)


# ==========================================
#  3. 两种条件生成
# ==========================================

def generate_from_first_line(model, first_line, stoi, itos, device,
                             strategy="temperature", temperature=0.8, top_k=10):
    """
    首句续写：给定第一句（7字），生成完整 28 字绝句
    """
    print(f"\n[首句续写] 输入: {first_line}")
    assert len(first_line) == 7, f"[首句续写] 首句长度必须为 7，实际为 {len(first_line)}"

    prompt_ids = [stoi[BOS_TOKEN]]
    prompt_ids.extend(stoi.get(ch, stoi[UNK_TOKEN]) for ch in first_line)

    generated_ids = generate_sequence(
        model=model,
        prompt_ids=prompt_ids,
        stoi=stoi,
        itos=itos,
        device=device,
        max_new_tokens=21,
        strategy=strategy,
        temperature=temperature,
        top_k=top_k,
        stop_at_eos=False,
    )

    poem_text = extract_text_from_ids(generated_ids, itos)[:28]
    print(f"[首句续写] 输出: {format_poem_for_display(poem_text)}")
    return poem_text


def generate_acrostic(model, head_chars, stoi, itos, device,
                      strategy="temperature", temperature=0.8, top_k=10):
    """
    藏头诗：给定 4 个字，分别作为四句首字，每句生成 7 字
    """
    print(f"\n[藏头诗] 输入: {head_chars}")
    assert len(head_chars) == 4, f"[藏头诗] 藏头字数量必须为 4，实际为 {len(head_chars)}"

    lines = []
    for line_idx, head_char in enumerate(head_chars, start=1):
        prompt_ids = [stoi[BOS_TOKEN], stoi.get(head_char, stoi[UNK_TOKEN])]

        generated_ids = generate_sequence(
            model=model,
            prompt_ids=prompt_ids,
            stoi=stoi,
            itos=itos,
            device=device,
            max_new_tokens=6,
            strategy=strategy,
            temperature=temperature,
            top_k=top_k,
            stop_at_eos=False,
        )

        line_text = extract_text_from_ids(generated_ids, itos)[:7]
        if len(line_text) < 7:
            line_text = line_text.ljust(7, head_char)
        lines.append(line_text)

    poem_text = "".join(lines)[:28]
    print(f"[藏头诗] 输出: {format_poem_for_display(poem_text)}")
    return poem_text


# ==========================================
#  4. 单次生成结果保存
# ==========================================

def save_single_generation_result(file_path, first_line_input, first_line_poem,
                                  acrostic_input, acrostic_poem, strategy, temperature, top_k):
    """
    保存单次生成结果到文本文件
    """
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("诗词生成结果\n")
        f.write("=" * 60 + "\n")
        f.write(f"strategy={strategy}, temperature={temperature}, top_k={top_k}\n")
        f.write("-" * 60 + "\n")
        f.write(f"[首句续写输入] {first_line_input}\n")
        f.write(f"[首句续写输出] {format_poem_for_display(first_line_poem)}\n")
        f.write("-" * 60 + "\n")
        f.write(f"[藏头诗输入] {acrostic_input}\n")
        f.write(f"[藏头诗输出] {format_poem_for_display(acrostic_poem)}\n")

    print(f"[输出] 单次生成结果已保存到: {file_path}")


def resolve_generation_inputs(default_first_line, default_acrostic):
    """
    处理终端输入，允许用户直接回车使用默认值

    参数:
        default_first_line: 默认首句
        default_acrostic:   默认藏头字

    返回:
        first_line_input: 首句续写输入
        acrostic_input:   藏头诗输入
    """
    print("\n" + "-" * 55)
    print(" 3. 终端输入设置")
    print("-" * 55)
    print(f"[默认首句] {default_first_line}")
    print(f"[默认藏头] {default_acrostic}")

    first_line_input = input("请输入首句续写的第一句（默认：春风又绿江南岸）：").strip()
    acrostic_input = input("请输入 4 个藏头字（默认：春江花月）：").strip()

    if first_line_input == "":
        first_line_input = default_first_line
    if acrostic_input == "":
        acrostic_input = default_acrostic

    return first_line_input, acrostic_input


# ==========================================
#  5. 主函数
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="诗词生成最小生成闭环")
    parser.add_argument("--checkpoint", type=str, default=BEST_MODEL_SAVE_PATH, help="checkpoint 路径")
    parser.add_argument("--strategy", type=str, default="temperature", choices=["temperature", "top_k"], help="采样策略")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="temperature 参数")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="top-k 参数")
    parser.add_argument("--first_line", type=str, default="春风又绿江南岸", help="默认首句续写输入")
    parser.add_argument("--acrostic", type=str, default="春江花月", help="默认藏头诗输入")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def main():
    print("=" * 60)
    print(" 诗词生成最小生成闭环")
    print(" 条件生成：首句续写 / 藏头诗")
    print("=" * 60)

    args = parse_args()
    configure_torch_runtime()
    set_seed(args.seed)
    ensure_dir(OUTPUT_DIR)
    ensure_dir(SAMPLES_DIR)

    print("\n" + "-" * 55)
    print(" 1. 检测计算设备")
    print("-" * 55)
    device = get_device()
    print(f"使用设备: {device}")
    if device == "cuda":
        print(f"  GPU 型号: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  未检测到 GPU，使用 CPU 进行生成")
    print(f"  PyTorch 版本: {torch.__version__}")
    print(f"  项目目录: {BASE_DIR}")

    print("\n" + "-" * 55)
    print(" 2. 加载模型与词表")
    print("-" * 55)
    model, checkpoint, stoi, itos = load_model_from_checkpoint(args.checkpoint, device)

    # 防御性检查：特殊 token 必须存在
    for token in [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]:
        assert token in stoi, f"[主函数] checkpoint 中缺少特殊 token: {token}"

    _ = checkpoint  # 保留变量，便于后续扩展历史信息打印

    first_line_input, acrostic_input = resolve_generation_inputs(
        default_first_line=args.first_line,
        default_acrostic=args.acrostic,
    )

    print("\n" + "-" * 55)
    print(" 4. 单次生成结果")
    print("-" * 55)
    print(f"[生成参数] strategy={args.strategy}, temperature={args.temperature}, top_k={args.top_k}")

    first_line_poem = generate_from_first_line(
        model=model,
        stoi=stoi,
        itos=itos,
        device=device,
        strategy=args.strategy,
        temperature=args.temperature,
        top_k=args.top_k,
        first_line=first_line_input,
    )
    first_line_analysis = analyze_generated_poem(first_line_poem, expected_len=28)

    acrostic_poem = generate_acrostic(
        model=model,
        stoi=stoi,
        itos=itos,
        device=device,
        strategy=args.strategy,
        temperature=args.temperature,
        top_k=args.top_k,
        head_chars=acrostic_input,
    )
    acrostic_analysis = analyze_generated_poem(acrostic_poem, expected_len=28)

    print("\n" + "=" * 55)
    print(" [生成结果分析]")
    print("=" * 55)
    print(f"\n[首句续写] 输出: {format_poem_for_display(first_line_poem)}")
    print(f"[首句续写] 实际长度: {first_line_analysis['format']['actual_len']}")
    print(f"[首句续写] 是否 28 字合规: {first_line_analysis['format']['exact_match']}")
    print(f"[首句续写] 重复分数: {first_line_analysis['repeat_score']:.4f}")

    print(f"\n[藏头诗] 输出: {format_poem_for_display(acrostic_poem)}")
    print(f"[藏头诗] 实际长度: {acrostic_analysis['format']['actual_len']}")
    print(f"[藏头诗] 是否 28 字合规: {acrostic_analysis['format']['exact_match']}")
    print(f"[藏头诗] 重复分数: {acrostic_analysis['repeat_score']:.4f}")

    save_name = f"single_generation_{args.strategy}_t{str(args.temperature).replace('.', '')}_k{args.top_k}.txt"
    save_path = os.path.join(SAMPLES_DIR, save_name)
    save_single_generation_result(
        file_path=save_path,
        first_line_input=first_line_input,
        first_line_poem=first_line_poem,
        acrostic_input=acrostic_input,
        acrostic_poem=acrostic_poem,
        strategy=args.strategy,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print("\n" + "=" * 60)
    print(" 诗词生成最小生成闭环完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
