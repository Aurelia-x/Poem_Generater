from collections import Counter

try:
    import pypinyin
    _HAS_PYPINYIN = True
except ImportError:
    _HAS_PYPINYIN = False


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


# ==========================================
#  2. 批量多样性指标
# ==========================================

def batch_diversity(poems):
    """
    统计一批诗歌的多样性，返回 3 个核心指标

    指标说明:
        uniq_rate:  样本去重率。N 首中有几首不重复，1.0=全部不同
        repeat_pct: 诗内平均重复度(%)。某 2-gram 最多出现次数/全诗 2-gram 总数,
                    例如「花花花花」repeat_pct 很高，越低越好
        inter_dens: 跨样本 2-gram 密度。所有样本的 2-gram 总数 / 去重后数量,
                    1.0=完全多样, 1.3=30%的 2-gram 在样本间重复, 越低越好

    返回: uniq_rate, repeat_pct, inter_dens
    """
    assert len(poems) > 0, "poems 不能为空"

    # 1. 样本去重率
    uniq_rate = len(set(poems)) / len(poems)

    # 2. 诗内平均重复度 → 百分比
    repeat_scores = []
    for p in poems:
        sc = repetition_score(p)
        repeat_scores.append(sc)
    repeat_pct = sum(repeat_scores) / len(repeat_scores) * 100

    # 3. 跨样本 2-gram 密度: total / distinct, 1.0 最理想
    all_ngrams = []
    for p in poems:
        if len(p) >= 2:
            all_ngrams.extend(p[i:i + 2] for i in range(len(p) - 1))
    total_n = len(all_ngrams)
    distinct_n = len(set(all_ngrams))
    inter_dens = total_n / distinct_n if distinct_n > 0 else 0.0

    return {
        "uniq_rate": uniq_rate,
        "repeat_pct": repeat_pct,
        "inter_dens": inter_dens,
    }


# ==========================================
#  3. 韵律评测 (pypinyin)
# ==========================================

def _extract_rhyme_final(char):
    """提取单个汉字的韵母（去声调），如「江」→ 'iang'"""
    if not _HAS_PYPINYIN:
        return None
    try:
        finals = pypinyin.lazy_pinyin(char, style=pypinyin.Style.FINALS, strict=False)
        return finals[0] if finals and finals[0] else None
    except Exception:
        return None


def check_jueju_rhyme(poem_text):
    """
    检查七言绝句押韵

    七言绝句 4 句，每句 7 字，句末字在下标 6, 13, 20, 27
    规则:
      - 宽松: 第2、4句末字押韵
      - 严格: 第1、2、4句末字押韵 (第3句通常不押)

    参数:
        poem_text: 28 字诗歌字符串

    返回:
        dict: end_chars(4个句末字), rhymes(4个韵母),
              rule_2_4(bool), rule_1_2_4(bool)
    """
    if not _HAS_PYPINYIN or len(poem_text) < 28:
        return {"end_chars": [], "rhymes": [], "rule_2_4": False, "rule_1_2_4": False, "error": "no_pypinyin_or_too_short"}

    end_positions = [6, 13, 20, 27]
    end_chars = [poem_text[i] for i in end_positions]
    rhymes = [_extract_rhyme_final(c) for c in end_chars]

    # 任一韵母为 None 则无法判定
    if any(r is None for r in rhymes):
        rule_24 = rule_124 = False
    else:
        rule_24 = (rhymes[1] == rhymes[3])   # 第2句 vs 第4句
        rule_124 = (rhymes[0] == rhymes[1] == rhymes[3])

    return {
        "end_chars": end_chars,
        "rhymes": rhymes,
        "rule_2_4": rule_24,
        "rule_1_2_4": rule_124,
    }


def batch_rhyme_rate(poems):
    """
    统计一批诗歌的押韵率

    返回:
        rate_2_4:   第2、4句押韵率
        rate_1_2_4: 第1、2、4句押韵率
        details:    每首的详情列表
    """
    results = [check_jueju_rhyme(p) for p in poems]
    valid = [r for r in results if r.get("error") is None]
    n = len(valid)
    if n == 0:
        return {"rate_2_4": 0.0, "rate_1_2_4": 0.0, "valid_count": 0, "details": results}
    return {
        "rate_2_4": sum(r["rule_2_4"] for r in valid) / n,
        "rate_1_2_4": sum(r["rule_1_2_4"] for r in valid) / n,
        "valid_count": n,
        "details": results,
    }
