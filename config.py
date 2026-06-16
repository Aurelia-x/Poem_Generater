import os

# ==========================================
#  全局可调超参数设置
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RANDOM_SEED = 42                           # 随机种子，保证实验可复现

RAW_DATA_PATH = os.path.join(BASE_DIR, "chinese_poem.txt")   # 原始诗歌数据路径
CACHE_DIR = os.path.join(BASE_DIR, "cache")                  # 缓存目录
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")       # 模型保存目录
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")               # 输出结果目录
SAMPLES_DIR = os.path.join(OUTPUT_DIR, "samples")            # 生成样例输出目录
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")            # 指标结果输出目录
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")            # 训练可视化输出目录
ABLATION_DIR = os.path.join(OUTPUT_DIR, "ablation")          # 消融实验输出目录
ABLATION_MANIFEST_PATH = os.path.join(ABLATION_DIR, "ablation_manifest.json")  # 消融实验模型清单

CLEAN_DATA_PATH = os.path.join(CACHE_DIR, "poems_clean.txt")         # 清洗后文本缓存
VOCAB_PATH = os.path.join(CACHE_DIR, "vocab.json")                   # 词表缓存
BEST_MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "poem_lstm_final_best.pth")   # 最佳模型保存路径
LAST_MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "poem_lstm_final_last.pth")   # 最终模型保存路径

MAX_POEMS = None                             # 最多使用多少首诗（None 表示全量）
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
MIN_CHAR_FREQ = 5                           # 全量训练时适当提高词频阈值，缩小词表并提升稳定性

# ==========================================
#  最终版主模型参数
# ==========================================

MODEL_TYPE = "lstm"                         # 最终版主模型类型
EMBED_DIM = 256                             # 最终版字符嵌入维度
HIDDEN_DIM = 512                            # 最终版隐藏状态维度
NUM_LAYERS = 2                              # 最终版使用双层 LSTM
DROPOUT = 0.3                               # 双层 LSTM 层间 dropout

BATCH_SIZE = 96                             # 全量训练时适当增大 batch，提高吞吐
LEARNING_RATE = 3e-4                        # 初始学习率
NUM_EPOCHS = 50                             # 全量数据下每轮代价更高，减少最大轮数并交给早停控制
GRAD_CLIP = 1.0                             # 梯度裁剪阈值
WEIGHT_DECAY = 5e-5                         # 全量数据下适当减弱正则，避免欠拟合
LOG_EVERY_N_BATCHES = 0                     # 0 表示关闭 batch 内日志，只保留 epoch 汇总
SCHEDULER_FACTOR = 0.5                      # 验证集指标停滞时学习率衰减系数
SCHEDULER_PATIENCE = 1                      # 学习率调度器等待轮数
SCHEDULER_MIN_LR = 1e-5                     # 学习率下限
EARLY_STOPPING_PATIENCE = 3                 # 全量训练时更早停止，节省时间
EARLY_STOPPING_MIN_DELTA = 0.1             # PPL 改善低于此阈值视为无改善

SKIP_TRAINING = True                       # True=跳过训练，直接加载本地模型参数

GEN_MAX_STEPS = 40                          # 单次生成的最大步数上限，防止死循环
DEFAULT_TEMPERATURE = 0.8                   # 默认 temperature
DEFAULT_TOP_K = 0                            # 默认 top-k (0=全词表 temperature 采样)


# ==========================================
#  配置辅助函数
# ==========================================

def build_runtime_config():
    """
    构建当前默认主实验配置字典

    返回:
        runtime_config: 用于训练与保存 checkpoint 的配置字典
    """
    return {
        "experiment_name": "poem_lstm_final",
        "model_type": MODEL_TYPE,
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "grad_clip": GRAD_CLIP,
        "weight_decay": WEIGHT_DECAY,
        "log_every_n_batches": LOG_EVERY_N_BATCHES,
        "scheduler_factor": SCHEDULER_FACTOR,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "scheduler_min_lr": SCHEDULER_MIN_LR,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "random_seed": RANDOM_SEED,
        "best_model_save_path": BEST_MODEL_SAVE_PATH,
        "last_model_save_path": LAST_MODEL_SAVE_PATH,
    }


ABLATION_EXPERIMENTS = [
    {
        "experiment_name": "ablation_a_baseline",
        "model_type": "lstm",
        "embed_dim": 128,
        "hidden_dim": 256,
        "num_layers": 1,
        "dropout": 0.0,
        "learning_rate": LEARNING_RATE,
        "num_epochs": 20,
        "grad_clip": GRAD_CLIP,
        "weight_decay": 0.0,
        "log_every_n_batches": LOG_EVERY_N_BATCHES,
        "scheduler_factor": SCHEDULER_FACTOR,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "scheduler_min_lr": SCHEDULER_MIN_LR,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "best_model_save_path": os.path.join(CHECKPOINT_DIR, "ablation_a_baseline_best.pth"),
        "last_model_save_path": os.path.join(CHECKPOINT_DIR, "ablation_a_baseline_last.pth"),
    },
    {
        "experiment_name": "ablation_b_deeper",
        "model_type": "lstm",
        "embed_dim": 128,
        "hidden_dim": 256,
        "num_layers": 2,
        "dropout": 0.3,
        "learning_rate": LEARNING_RATE,
        "num_epochs": 20,
        "grad_clip": GRAD_CLIP,
        "weight_decay": 0.0,
        "log_every_n_batches": LOG_EVERY_N_BATCHES,
        "scheduler_factor": SCHEDULER_FACTOR,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "scheduler_min_lr": SCHEDULER_MIN_LR,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "best_model_save_path": os.path.join(CHECKPOINT_DIR, "ablation_b_deeper_best.pth"),
        "last_model_save_path": os.path.join(CHECKPOINT_DIR, "ablation_b_deeper_last.pth"),
    },
    {
        "experiment_name": "ablation_c_wider",
        "model_type": "lstm",
        "embed_dim": 256,
        "hidden_dim": 512,
        "num_layers": 1,
        "dropout": 0.0,
        "learning_rate": LEARNING_RATE,
        "num_epochs": 20,
        "grad_clip": GRAD_CLIP,
        "weight_decay": 0.0,
        "log_every_n_batches": LOG_EVERY_N_BATCHES,
        "scheduler_factor": SCHEDULER_FACTOR,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "scheduler_min_lr": SCHEDULER_MIN_LR,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "best_model_save_path": os.path.join(CHECKPOINT_DIR, "ablation_c_wider_best.pth"),
        "last_model_save_path": os.path.join(CHECKPOINT_DIR, "ablation_c_wider_last.pth"),
    },
    {
        "experiment_name": "ablation_d_regularized",
        "model_type": "lstm",
        "embed_dim": 128,
        "hidden_dim": 256,
        "num_layers": 1,
        "dropout": 0.0,
        "weight_decay": 1e-4,
        "learning_rate": LEARNING_RATE,
        "num_epochs": 20,
        "grad_clip": GRAD_CLIP,
        "log_every_n_batches": LOG_EVERY_N_BATCHES,
        "scheduler_factor": SCHEDULER_FACTOR,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "scheduler_min_lr": SCHEDULER_MIN_LR,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "best_model_save_path": os.path.join(CHECKPOINT_DIR, "ablation_d_regularized_best.pth"),
        "last_model_save_path": os.path.join(CHECKPOINT_DIR, "ablation_d_regularized_last.pth"),
    },
    {
        "experiment_name": "ablation_e_full",
        "model_type": "lstm",
        "embed_dim": 256,
        "hidden_dim": 512,
        "num_layers": 2,
        "dropout": 0.3,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "grad_clip": GRAD_CLIP,
        "weight_decay": WEIGHT_DECAY,
        "log_every_n_batches": LOG_EVERY_N_BATCHES,
        "scheduler_factor": SCHEDULER_FACTOR,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "scheduler_min_lr": SCHEDULER_MIN_LR,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "best_model_save_path": os.path.join(CHECKPOINT_DIR, "ablation_e_full_best.pth"),
        "last_model_save_path": os.path.join(CHECKPOINT_DIR, "ablation_e_full_last.pth"),
    },
    {
        "experiment_name": "ablation_f_gru",
        "model_type": "gru",
        "embed_dim": 128,
        "hidden_dim": 256,
        "num_layers": 1,
        "dropout": 0.0,
        "learning_rate": LEARNING_RATE,
        "num_epochs": 20,
        "grad_clip": GRAD_CLIP,
        "weight_decay": 0.0,
        "log_every_n_batches": LOG_EVERY_N_BATCHES,
        "scheduler_factor": SCHEDULER_FACTOR,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "scheduler_min_lr": SCHEDULER_MIN_LR,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "best_model_save_path": os.path.join(CHECKPOINT_DIR, "ablation_f_gru_best.pth"),
        "last_model_save_path": os.path.join(CHECKPOINT_DIR, "ablation_f_gru_last.pth"),
    },
]
