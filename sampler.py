import torch


# ==========================================
#  1. 采样策略
# ==========================================

def sample_with_temperature(logits, temperature=1.0):
    """
    基于 temperature 的随机采样

    参数:
        logits:       当前时间步输出 logits, 形状 (V,) 或 (1, V)
        temperature:  温度系数，越小越保守，越大越随机

    返回:
        next_id: 采样得到的下一个 token id
    """
    if logits.dim() == 2:
        logits = logits.squeeze(0)

    assert logits.dim() == 1, \
        f"[sample_with_temperature] logits 维度异常: 期望 1 维, 实际 {tuple(logits.shape)}"
    assert temperature > 0, "[sample_with_temperature] temperature 必须大于 0"

    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1).item()
    return next_id


def sample_with_top_k(logits, top_k=10, temperature=1.0):
    """
    基于 top-k 的随机采样

    参数:
        logits:       当前时间步输出 logits, 形状 (V,) 或 (1, V)
        top_k:        仅从概率最高的前 k 个 token 中采样
        temperature:  温度系数

    返回:
        next_id: 采样得到的下一个 token id
    """
    if logits.dim() == 2:
        logits = logits.squeeze(0)

    assert logits.dim() == 1, \
        f"[sample_with_top_k] logits 维度异常: 期望 1 维, 实际 {tuple(logits.shape)}"
    assert top_k > 0, "[sample_with_top_k] top_k 必须大于 0"
    assert temperature > 0, "[sample_with_top_k] temperature 必须大于 0"

    scaled_logits = logits / temperature
    k = min(top_k, scaled_logits.shape[0])
    top_k_values, top_k_indices = torch.topk(scaled_logits, k)

    filtered_logits = torch.full_like(scaled_logits, float("-inf"))
    filtered_logits.scatter_(0, top_k_indices, top_k_values)

    probs = torch.softmax(filtered_logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1).item()
    return next_id


def sample_next_token(logits, strategy="temperature", temperature=1.0, top_k=10):
    """
    统一采样入口

    参数:
        logits:       当前时间步输出 logits
        strategy:     采样策略，"temperature" 或 "top_k"
        temperature:  温度系数
        top_k:        top-k 参数

    返回:
        next_id: 下一个 token id
    """
    strategy = strategy.lower()

    if strategy == "temperature":
        return sample_with_temperature(logits, temperature=temperature)
    if strategy == "top_k":
        return sample_with_top_k(logits, top_k=top_k, temperature=temperature)

    raise ValueError(f"[sample_next_token] 不支持的采样策略: {strategy}")
