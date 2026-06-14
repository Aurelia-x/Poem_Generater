import torch
import torch.nn as nn


# ==========================================
#  1. 诗歌生成模型
# ==========================================

class PoemLanguageModel(nn.Module):
    """
    字符级诗歌语言模型

    模型结构:
        1. Embedding(V, E): 字符索引 -> 稠密向量
        2. LSTM / GRU:      建模字符序列上下文
        3. Linear(H, V):    每个时间步预测下一个字符

    输入:
        x: (B, T)

    输出:
        logits: (B, T, V)
        hidden: RNN 最终隐藏状态
    """

    def __init__(self, vocab_size, model_type="lstm",
                 embed_dim=128, hidden_dim=256,
                 num_layers=1, dropout=0.0):
        super(PoemLanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.model_type = model_type.lower()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self._shape_checked = False

        print(f"\n[PoemLanguageModel __init__] 开始构建诗歌生成模型...")
        print(f"  [PoemModel] 词表大小 V = {vocab_size}")
        print(f"  [PoemModel] 模型类型   = {self.model_type.upper()}")
        print(f"  [PoemModel] 嵌入维度 E = {embed_dim}")
        print(f"  [PoemModel] 隐藏维度 H = {hidden_dim}")
        print(f"  [PoemModel] 层数       = {num_layers}")
        print(f"  [PoemModel] dropout    = {dropout}")

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        print(f"  [PoemModel] Embedding 构建完成: nn.Embedding({vocab_size}, {embed_dim})")

        rnn_dropout = dropout if num_layers > 1 else 0.0
        if self.model_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        elif self.model_type == "gru":
            self.rnn = nn.GRU(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        else:
            raise ValueError(f"[PoemModel] 不支持的模型类型: {model_type}")

        print(f"  [PoemModel] {self.model_type.upper()} 主体构建完成")

        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        print(f"  [PoemModel] 输出层构建完成: nn.Linear({hidden_dim}, {vocab_size})")
        print(f"[PoemLanguageModel __init__] 诗歌生成模型构建完成")

    def forward(self, x, hidden=None):
        """
        前向传播

        参数:
            x:      输入字符索引, 形状 (B, T)
            hidden: 上一时刻隐藏状态，训练时一般为 None

        返回:
            logits: (B, T, V)
            hidden: RNN 最终隐藏状态
        """
        B, T = x.shape

        x_emb = self.embedding(x)  # (B, T) -> (B, T, E)
        assert x_emb.shape == (B, T, self.embed_dim), \
            f"[PoemModel forward] Embedding 输出形状异常: 期望 ({B},{T},{self.embed_dim}), 实际 {tuple(x_emb.shape)}"

        rnn_out, hidden = self.rnn(x_emb, hidden)  # (B, T, H)
        assert rnn_out.shape == (B, T, self.hidden_dim), \
            f"[PoemModel forward] RNN 输出形状异常: 期望 ({B},{T},{self.hidden_dim}), 实际 {tuple(rnn_out.shape)}"

        logits = self.output_layer(rnn_out)  # (B, T, V)
        assert logits.shape == (B, T, self.vocab_size), \
            f"[PoemModel forward] logits 形状异常: 期望 ({B},{T},{self.vocab_size}), 实际 {tuple(logits.shape)}"

        if not self._shape_checked:
            print(f"  [PoemModel forward] shape 断言通过: "
                  f"x ({B},{T}) -> emb ({B},{T},{self.embed_dim}) "
                  f"-> rnn_out ({B},{T},{self.hidden_dim}) "
                  f"-> logits ({B},{T},{self.vocab_size})")
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
