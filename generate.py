import argparse
import os

import torch

try:
    import pypinyin
    _HAS_PYPINYIN = True
except ImportError:
    _HAS_PYPINYIN = False

from config import (
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
#  0. 韵律工具
# ==========================================

def _extract_final(char):
    """提取汉字韵母，如 '江' → 'iang'"""
    if not _HAS_PYPINYIN:
        return None
    try:
        finals = pypinyin.lazy_pinyin(char, style=pypinyin.Style.FINALS, strict=False)
        return finals[0] if finals and finals[0] else None
    except Exception:
        return None


def build_rhyme_vocab_map(itos):
    """
    构建韵母 → 字符 id 列表的映射

    参数:
        itos: id → 字符

    返回:
        {韵母: [id, id, ...]}
    """
    if not _HAS_PYPINYIN:
        return {}
    rhyme_map = {}
    for idx, char in enumerate(itos):
        if char in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN):
            continue
        final = _extract_final(char)
        if final:
            rhyme_map.setdefault(final, []).append(idx)
    return rhyme_map


# ==========================================
#  1. 加载模型与词表
# ==========================================

def load_model_from_checkpoint(checkpoint_path, device, verbose=True):
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

    if verbose:
        print(f"[模型加载] epoch={checkpoint.get('epoch', 'N/A')} best_val_ppl={checkpoint.get('best_val_ppl', 'N/A')} vocab_size={len(itos)}")

    return model, checkpoint, stoi, itos


# ==========================================
#  2. 基础逐字生成函数
# ==========================================

@torch.no_grad()
def generate_sequence(model, prompt_ids, stoi, itos, device,
                      max_new_tokens=28, temperature=0.8, top_k=0,
                      stop_at_eos=True,
                      rhyme_boost=None):
    """
    从 prompt 开始，逐字生成后续 token

    top_k=0 表示全词表 temperature 采样；top_k>0 表示 top-k 截断 + temperature 采样

    rhyme_boost: (rhyme_map, target_final, boost_positions) 或 None
      在 boost_positions 位置，提高 target_final 同韵字的 logits
    """
    generated_ids = list(prompt_ids)

    for step in range(max_new_tokens):
        input_tensor = torch.tensor([generated_ids], dtype=torch.long, device=device)
        logits, _ = model(input_tensor)
        next_logits = logits[0, -1, :].clone()

        # 韵律引导：在指定位置提高同韵字概率
        if rhyme_boost is not None:
            rhyme_map, target_final, boost_positions = rhyme_boost
            if step in boost_positions and target_final in rhyme_map:
                rhyme_ids = rhyme_map[target_final]
                next_logits[rhyme_ids] += 3.0

        next_id = sample_next_token(next_logits, temperature=temperature, top_k=top_k if top_k > 0 else None)

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
                             temperature=0.8, top_k=0, rhyme_map=None):
    """
    首句续写：给定第一句（7字），生成完整 28 字绝句

    rhyme_map: {韵母: [char_id]} or None → 若提供则做韵律引导
    """
    assert len(first_line) == 7, f"[首句续写] 首句长度必须为 7，实际为 {len(first_line)}"

    prompt_ids = [stoi[BOS_TOKEN]]
    prompt_ids.extend(stoi.get(ch, stoi[UNK_TOKEN]) for ch in first_line)

    # 构建韵律引导
    boost = None
    if rhyme_map and _HAS_PYPINYIN:
        target = _extract_final(first_line[-1])
        if target and target in rhyme_map:
            # 生成 21 个 token: step 6=第2句末, step 20=第4句末
            boost = (rhyme_map, target, [6, 20])

    generated_ids = generate_sequence(
        model=model,
        prompt_ids=prompt_ids,
        stoi=stoi,
        itos=itos,
        device=device,
        max_new_tokens=21,
        temperature=temperature,
        top_k=top_k,
        stop_at_eos=False,
        rhyme_boost=boost,
    )

    poem_text = extract_text_from_ids(generated_ids, itos)[:28]
    return poem_text


def generate_acrostic(model, head_chars, stoi, itos, device,
                      temperature=0.8, top_k=0, rhyme_map=None):
    """
    藏头诗：给定 4 个字，分别作为四句首字，每句生成 7 字

    rhyme_map: {韵母: [char_id]} or None → 若提供，以第1句韵母引导第2、4句
    """
    assert len(head_chars) == 4, f"[藏头诗] 藏头字数量必须为 4，实际为 {len(head_chars)}"

    lines = []
    target_rhyme = None

    for line_idx, head_char in enumerate(head_chars, start=1):
        prompt_ids = [stoi[BOS_TOKEN], stoi.get(head_char, stoi[UNK_TOKEN])]

        # 韵律引导：第2、4句末字押第1句韵
        boost = None
        if rhyme_map and _HAS_PYPINYIN and target_rhyme and target_rhyme in rhyme_map and line_idx in (2, 4):
            boost = (rhyme_map, target_rhyme, [5])  # step 5 = 句末

        generated_ids = generate_sequence(
            model=model,
            prompt_ids=prompt_ids,
            stoi=stoi,
            itos=itos,
            device=device,
            max_new_tokens=6,
            temperature=temperature,
            top_k=top_k,
            stop_at_eos=False,
            rhyme_boost=boost,
        )

        line_text = extract_text_from_ids(generated_ids, itos)[:7]
        if len(line_text) < 7:
            line_text = line_text.ljust(7, head_char)
        lines.append(line_text)

        # 第1句生成后提取韵母
        if rhyme_map and _HAS_PYPINYIN and line_idx == 1:
            target_rhyme = _extract_final(line_text[-1])

    poem_text = "".join(lines)[:28]
    return poem_text


# ==========================================
#  4. 单次生成结果保存
# ==========================================

def save_single_generation_result(file_path, first_line_input, first_line_poem,
                                  acrostic_input, acrostic_poem, temperature, top_k):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("诗词生成结果\n")
        f.write("=" * 60 + "\n")
        f.write(f"temperature={temperature}, top_k={top_k}\n")
        f.write("-" * 60 + "\n")
        f.write(f"[首句续写输入] {first_line_input}\n")
        f.write(f"[首句续写输出] {format_poem_for_display(first_line_poem)}\n")
        f.write("-" * 60 + "\n")
        f.write(f"[藏头诗输入] {acrostic_input}\n")
        f.write(f"[藏头诗输出] {format_poem_for_display(acrostic_poem)}\n")


def resolve_generation_inputs(default_first_line, default_acrostic,
                             default_temperature, default_top_k,
                             default_rhyme=True):
    """
    终端交互输入，回车保留默认值

    返回:
        first_line_input, acrostic_input, temperature, top_k, use_rhyme
    """
    rhyme_label = "ON" if default_rhyme else "OFF"
    print(f"\n{'=' * 44}")
    print(" 交互设置 (直接回车使用默认值)")
    print(f"{'=' * 44}")
    print(f"[默认首句] {default_first_line}")
    print(f"[默认藏头] {default_acrostic}")
    print(f"[默认参数] temperature={default_temperature} top_k={default_top_k} rhyme_boost={rhyme_label}")

    first_line_input = input("首句续写第一句: ").strip()
    acrostic_input = input("4个藏头字: ").strip()

    temperature_input = input(f"temperature: ").strip()
    top_k_input = input(f"top_k (0=全词表): ").strip()
    rhyme_input = input(f"韵律引导 (on/off, 默认{rhyme_label}): ").strip()

    if first_line_input == "":
        first_line_input = default_first_line
    if acrostic_input == "":
        acrostic_input = default_acrostic

    temperature = default_temperature if temperature_input == "" else float(temperature_input)
    top_k = default_top_k if top_k_input == "" else int(top_k_input)

    if rhyme_input == "":
        use_rhyme = default_rhyme
    else:
        use_rhyme = rhyme_input.lower() in ("on", "yes", "y", "1", "true")

    return first_line_input, acrostic_input, temperature, top_k, use_rhyme


# ==========================================
#  5. 主函数
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="诗词生成最小生成闭环")
    parser.add_argument("--checkpoint", type=str, default=BEST_MODEL_SAVE_PATH, help="checkpoint 路径")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="temperature 参数")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="top-k 参数 (0=全词表)")
    parser.add_argument("--first_line", type=str, default="春风又绿江南岸", help="默认首句续写输入")
    parser.add_argument("--acrostic", type=str, default="春江花月", help="默认藏头诗输入")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--rhyme", action="store_true", default=True, help="启用韵律引导（默认开启）")
    parser.add_argument("--no_rhyme", action="store_true", default=False, help="禁用韵律引导")
    return parser.parse_args()


def generate_and_display(model, stoi, itos, device, first_line_input, acrostic_input,
                         temperature, top_k, rhyme_map):
    """
    执行一次生成并打印结果
    """
    rhyme_status = "ON" if rhyme_map else "OFF"
    print(f"\n[生成参数] temperature={temperature} top_k={top_k} rhyme_boost={rhyme_status}")

    first_line_poem = generate_from_first_line(
        model=model, stoi=stoi, itos=itos, device=device,
        temperature=temperature, top_k=top_k,
        first_line=first_line_input,
        rhyme_map=rhyme_map,
    )
    first_line_analysis = analyze_generated_poem(first_line_poem, expected_len=28)

    acrostic_poem = generate_acrostic(
        model=model, stoi=stoi, itos=itos, device=device,
        temperature=temperature, top_k=top_k,
        head_chars=acrostic_input,
        rhyme_map=rhyme_map,
    )
    acrostic_analysis = analyze_generated_poem(acrostic_poem, expected_len=28)

    print(f"[首句续写] {format_poem_for_display(first_line_poem)} len={first_line_analysis['format']['actual_len']} repeat={first_line_analysis['repeat_score']:.4f}")
    print(f"[藏头诗]   {format_poem_for_display(acrostic_poem)} len={acrostic_analysis['format']['actual_len']} repeat={acrostic_analysis['repeat_score']:.4f}")

    return first_line_poem, first_line_analysis, acrostic_poem, acrostic_analysis


def main():
    print("=" * 50)
    print(" 诗词生成 | 首句续写 / 藏头诗")
    print("=" * 50)

    args = parse_args()
    configure_torch_runtime()
    set_seed(args.seed)
    ensure_dir(OUTPUT_DIR)
    ensure_dir(SAMPLES_DIR)

    device = get_device()
    if device == "cuda":
        print(f"[设备] CUDA | GPU: {torch.cuda.get_device_name(0)} | PyTorch: {torch.__version__}")
    else:
        print(f"[设备] CPU | PyTorch: {torch.__version__}")

    model, checkpoint, stoi, itos = load_model_from_checkpoint(args.checkpoint, device)

    for token in [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]:
        assert token in stoi, f"[主函数] checkpoint 中缺少特殊 token: {token}"

    _ = checkpoint

    # 韵律引导
    use_rhyme = args.rhyme and not args.no_rhyme
    if use_rhyme:
        rhyme_map = build_rhyme_vocab_map(itos)
        if rhyme_map:
            print(f"[韵律] 已构建韵母映射, {len(rhyme_map)} 个韵母类别")
        else:
            print("[韵律] pypinyin 未安装，韵律引导不可用")
    else:
        rhyme_map = None
        print("[韵律] 已禁用")

    first_line_input, acrostic_input, temperature, top_k, use_rhyme = resolve_generation_inputs(
        default_first_line=args.first_line,
        default_acrostic=args.acrostic,
        default_temperature=args.temperature,
        default_top_k=args.top_k,
        default_rhyme=args.rhyme and not args.no_rhyme,
    )

    generate_and_display(
        model, stoi, itos, device,
        first_line_input, acrostic_input,
        temperature, top_k,
        rhyme_map if use_rhyme else None,
    )

    while True:
        print()
        choice = input("[q 重新生成 / 回车退出]: ").strip()
        if choice == "":
            break
        if choice.lower() == "q":
            first_line_input, acrostic_input, temperature, top_k, use_rhyme = resolve_generation_inputs(
                default_first_line=first_line_input,
                default_acrostic=acrostic_input,
                default_temperature=temperature,
                default_top_k=top_k,
                default_rhyme=use_rhyme,
            )
            generate_and_display(
                model, stoi, itos, device,
                first_line_input, acrostic_input,
                temperature, top_k,
                rhyme_map if use_rhyme else None,
            )

    print("\n生成结束")


if __name__ == "__main__":
    main()
