import torch
import torch.nn as nn


# ==========================================
#  1. 诗歌生成模型
# ==========================================

class PoemLanguageModel(nn.Module):
    """
    字符级诗歌语言模型

    ================================================================================
    x, y 来源
    ================================================================================
    来自 data_utils.PoemDataset，每首 28 字七言绝句的处理:

    原始诗歌: "春风又绿江南岸..." (28 个汉字)
    编码后:   [BOS] + [char_ids × 28] + [EOS] = 长度 30 的整数序列 seq

    数据集构造 (自回归语言模型):
        x = seq[:-1]  →  [BOS, char_1, char_2, ..., char_28]  (前 29 个 token)
        y = seq[1:]   →  [char_1, char_2, ..., char_28, EOS]  (后 29 个 token)
        x 形状: (B, T=29)
        y 形状: (B, T=29)

    含义: 模型读 x，预测 y。即用"<BOS>"预测第一个字，用第一个字预测第二个字，...
    B=批次大小, T=29 (序列长度), V=词表大小

    ================================================================================
    模型结构 & 维度变化
    ================================================================================
    1. Embedding(V, E):   字符索引 → 稠密向量
       输入: (B, T)        → 输出: (B, T, E)
       参数量: V × E

    2. LSTM / GRU:        建模字符序列上下文
       输入: (B, T, E)     → 输出: (B, T, H)
       LSTM 参数量: 4 × H × (E + H + 2) × num_layers  (4个门: 输入/遗忘/输出/候选)
       GRU  参数量: 3 × H × (E + H + 2) × num_layers  (3个门: 重置/更新/新记忆)
       多层时: 第 1 层输入 E, 后续层输入 H (输出也 H)

    3. Linear(H, V):      每个时间步预测下一个字符 (时间步共享)
       输入: (B, T, H)     → 输出: (B, T, V) = logits
       参数量: H × V + V

    ================================================================================
    损失计算 (在 trainer.py 中)
    ================================================================================
    logits: (B, T, V) → reshape → (B*T, V)
    targets: (B, T)   → reshape → (B*T,)   (整数标签, 0~V-1)
    CrossEntropyLoss(logits, targets) → 标量

    ================================================================================
    输入输出
    ================================================================================
    输入:
        x: (B, T)  整数序列

    输出:
        logits: (B, T, V)  每个位置对词表中所有字符的预测分数
        hidden: RNN 最终隐藏状态 (生成时用于逐个 token 自回归解码)
    """

    def __init__(self, vocab_size, model_type="lstm",
                 embed_dim=128, hidden_dim=256,
                 num_layers=1, dropout=0.0):
        super(PoemLanguageModel, self).__init__()
        self.vocab_size = vocab_size     # V: 词表中不重复字符的总数
        self.model_type = model_type.lower()
        self.embed_dim = embed_dim       # E: 嵌入向量维度
        self.hidden_dim = hidden_dim     # H: RNN 隐藏状态维度
        self.num_layers = num_layers     # RNN 堆叠层数
        self.dropout = dropout
        self._shape_checked = False

        print(f"[模型] {self.model_type.upper()} V={vocab_size} E={embed_dim} H={hidden_dim} layers={num_layers} dropout={dropout}")

        # Embedding: (V) → (E), 把每个字符 ID 映射为稠密向量
        # 参数量: V × E
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # RNN: batch_first=True 使输入形状为 (B, T, E) 而不是默认的 (T, B, E)
        # dropout: 仅当 num_layers > 1 时启用层间 dropout, 单层时 PyTorch 不允许
        rnn_dropout = dropout if num_layers > 1 else 0.0
        if self.model_type == "lstm":
            # LSTM 4 个门: forget_gate, input_gate, cell_gate, output_gate
            # 每个门: W_ih (H×E) + W_hh (H×H) + b_ih (H) + b_hh (H)
            # 单层参数量: 4 × (H×E + H×H + 2H) = 4H × (E + H + 2)
            self.rnn = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        elif self.model_type == "gru":
            # GRU 3 个门: reset_gate, update_gate, new_gate
            # 每个门: W_ih (H×E) + W_hh (H×H) + b_ih (H) + b_hh (H)
            # 单层参数量: 3 × (H×E + H×H + 2H) = 3H × (E + H + 2)
            self.rnn = nn.GRU(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        else:
            raise ValueError(f"[PoemModel] 不支持的模型类型: {model_type}")

        # 输出层: 把每个时间步的隐藏状态映射到词表大小的 logits
        # 参数量: H × V + V (weight + bias)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        前向传播

        维度变化总览 (以基线 LSTM-128 为例: V≈5000, E=128, H=128, T=29):
          x (B, 29) 整数 ID
            → Embedding 查表 → x_emb (B, 29, 128)      参数量: V×E ≈ 5000×128 = 640,000
            → LSTM 序列建模 → rnn_out (B, 29, 128)      参数量: 4×128×(128+128+2) = 132,096
            → Linear 分类   → logits (B, 29, 5000)      参数量: 128×5000+5000 = 645,000
        总参数量 ~ 1.4M

        参数:
            x:      (B, T) 输入字符索引序列, B=batch_size, T=序列长度(29)
            hidden: RNN 隐藏状态, None=初次前向(训练时), 传入=自回归逐字生成

        返回:
            logits: (B, T, V) 每个位置对 V 个字符的预测分数 (未归一化)
            hidden: RNN 最终隐藏状态, 用于下一步自回归生成
        """
        B, T = x.shape

        # ★ Step 1: Embedding — 把离散字符 ID 映射为稠密向量
        # (B, T) → 查 Embedding 表 → (B, T, E)
        # 每个 ID 查出一个长度为 E 的向量, 这 E 维是模型"学到的"字符语义表示
        x_emb = self.embedding(x)

        # ★ Step 2: RNN — 序列建模, 让每个位置的向量"看到"前文上下文
        # (B, T, E) → 通过 LSTM/GRU 逐时间步处理 → (B, T, H)
        # rnn_out[i] 是第 i 个时间步的隐藏状态, 蕴含了 ≤i 位置的全部上文信息
        # hidden 是最终隐藏状态 (LSTM 返回 (h_n, c_n), GRU 返回 h_n)
        rnn_out, hidden = self.rnn(x_emb, hidden)

        # ★ Step 3: Linear — 把隐藏状态映射为词表大小的预测分数
        # (B, T, H) → 全连接 → (B, T, V)
        # logits[b, t, :] 是第 b 个样本、第 t 个位置对所有 V 个字符的预测分
        # 取 argmax 得到预测字符, 取 softmax 得到概率分布
        logits = self.output_layer(rnn_out)

        if not self._shape_checked:
            assert x_emb.shape == (B, T, self.embed_dim), \
                f"[forward] Embedding 形状异常: 期望 ({B},{T},{self.embed_dim}), 实际 {tuple(x_emb.shape)}"
            assert rnn_out.shape == (B, T, self.hidden_dim), \
                f"[forward] RNN 形状异常: 期望 ({B},{T},{self.hidden_dim}), 实际 {tuple(rnn_out.shape)}"
            assert logits.shape == (B, T, self.vocab_size), \
                f"[forward] logits 形状异常: 期望 ({B},{T},{self.vocab_size}), 实际 {tuple(logits.shape)}"
            self._shape_checked = True

        return logits, hidden


# ==========================================
#  2. 构造模型工厂函数
# ==========================================

def build_model(vocab_size, model_type="lstm",
                embed_dim=128, hidden_dim=256,
                num_layers=1, dropout=0.0):
    """
    根据配置构建模型

    参数:
        vocab_size: 词表大小
        model_type: "lstm" 或 "gru"
        embed_dim:  嵌入维度
        hidden_dim: 隐藏维度
        num_layers: 层数
        dropout:    dropout

    返回:
        model: 构建好的诗歌语言模型
    """
    model = PoemLanguageModel(
        vocab_size=vocab_size,
        model_type=model_type,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    return model
