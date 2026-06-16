import torch


def sample_next_token(logits, temperature=1.0, top_k=None):
    """
    统一采样入口：temperature + 可选 top-k 截断

    参数:
        logits:      当前时间步输出 logits, 形状 (V,) 或 (1, V)
        temperature: 温度系数，越小越保守，越大越随机
        top_k:       top-k 截断值，None 或 0 表示不截断（全词表采样）

    返回:
        next_id: 采样得到的下一个 token id
    """
    if logits.dim() == 2:
        logits = logits.squeeze(0)

    assert logits.dim() == 1, f"logits 维度异常: {tuple(logits.shape)}"
    assert temperature > 0, f"temperature 必须 > 0, 实际 {temperature}"

    scaled_logits = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(top_k, scaled_logits.shape[0])
        top_k_values, top_k_indices = torch.topk(scaled_logits, k)
        filtered_logits = torch.full_like(scaled_logits, float("-inf"))
        filtered_logits.scatter_(0, top_k_indices, top_k_values)
        probs = torch.softmax(filtered_logits, dim=-1)
    else:
        probs = torch.softmax(scaled_logits, dim=-1)

    next_id = torch.multinomial(probs, num_samples=1).item()
    return next_id
