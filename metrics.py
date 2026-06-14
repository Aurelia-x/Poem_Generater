from collections import Counter


# ==========================================
#  1. 生成质量基础指标
# ==========================================

def format_compliance(text, expected_len=28):
    """
    计算单首诗歌的格式合规情况

    参数:
        text:         生成的诗歌文本
        expected_len: 期望字数，七言绝句为 28

    返回:
        result: 包含是否完全合规、实际长度、偏差等信息
    """
    actual_len = len(text)
    exact_match = (actual_len == expected_len)
    deviation = abs(actual_len - expected_len)
    relaxed_score = min(actual_len, expected_len) / expected_len

    return {
        "exact_match": exact_match,
        "actual_len": actual_len,
        "deviation": deviation,
        "relaxed_score": relaxed_score,
    }


def batch_format_compliance(poems, expected_len=28):
    """
    统计一批诗歌的格式合规率

    参数:
        poems:        诗歌字符串列表
        expected_len: 期望字数

    返回:
        summary: 批量统计结果
    """
    assert len(poems) > 0, "[batch_format_compliance] poems 不能为空"

    results = [format_compliance(poem, expected_len=expected_len) for poem in poems]
    exact_count = sum(item["exact_match"] for item in results)
    avg_relaxed_score = sum(item["relaxed_score"] for item in results) / len(results)

    return {
        "count": len(poems),
        "exact_match_count": exact_count,
        "exact_match_rate": exact_count / len(poems),
        "avg_relaxed_score": avg_relaxed_score,
        "details": results,
    }


def repetition_score(text):
    """
    粗略统计文本重复程度

    简化定义:
        统计所有长度为 2 的重复片段（2-gram）中，出现次数最多的重复程度

    参数:
        text: 输入文本

    返回:
        score: 重复分数，越高表示越可能重复
    """
    if len(text) < 4:
        return 0.0

    bi_grams = [text[i:i + 2] for i in range(len(text) - 1)]
    counter = Counter(bi_grams)
    max_repeat = max(counter.values())
    return max_repeat / len(bi_grams)


def analyze_generated_poem(text, expected_len=28):
    """
    对单首生成诗歌做基础分析

    参数:
        text:         生成文本
        expected_len: 期望长度

    返回:
        analysis: 基础分析结果
    """
    format_info = format_compliance(text, expected_len=expected_len)
    repeat_score = repetition_score(text)

    return {
        "text": text,
        "format": format_info,
        "repeat_score": repeat_score,
    }
