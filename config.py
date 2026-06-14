import os

# ==========================================
#  全局可调超参数设置
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RANDOM_SEED = 42                           # 随机种子，保证实验可复现

RAW_DATA_PATH = os.path.join(BASE_DIR, "chinese_poem.txt")   # 原始诗歌数据路径
CACHE_DIR = os.path.join(BASE_DIR, "cache")                  # 缓存目录
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")       # 模型保存目录

CLEAN_DATA_PATH = os.path.join(CACHE_DIR, "poems_clean.txt")         # 清洗后文本缓存
VOCAB_PATH = os.path.join(CACHE_DIR, "vocab.json")                   # 词表缓存
BEST_MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "poem_lstm_best.pth")   # 最佳模型保存路径
LAST_MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "poem_lstm_last.pth")   # 最终模型保存路径

MAX_POEMS = 5000                            # 最多使用多少首诗（None 表示全量）
TRAIN_RATIO = 0.8                           # 训练集比例
VAL_RATIO = 0.1                             # 验证集比例
TEST_RATIO = 0.1                            # 测试集比例

SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]   # 特殊 token 列表
PAD_TOKEN = "[PAD]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
UNK_TOKEN = "[UNK]"

POEM_CHAR_LEN = 28                          # 七言绝句总字数（4句×7字）
SEQ_INPUT_LEN = 29                          # 输入长度 = [BOS] + 28字
SEQ_TARGET_LEN = 29                         # 目标长度 = 28字 + [EOS]
MIN_CHAR_FREQ = 1                           # 字符最小词频，小于该值的字映射为 UNK

MODEL_TYPE = "lstm"                         # 最小训练闭环先使用 LSTM
EMBED_DIM = 128                             # 字符嵌入维度
HIDDEN_DIM = 256                            # 隐藏状态维度
NUM_LAYERS = 1                              # 最小训练闭环先使用单层
DROPOUT = 0.0                               # 单层 RNN 不使用 dropout

BATCH_SIZE = 64                             # 批量大小
LEARNING_RATE = 1e-3                        # 学习率
NUM_EPOCHS = 12                             # 训练轮数
GRAD_CLIP = 1.0                             # 梯度裁剪阈值
LOG_EVERY_N_BATCHES = 50                    # 每隔多少个 batch 打印一次训练日志

SKIP_TRAINING = False                       # True=跳过训练，直接加载本地模型参数
