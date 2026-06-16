# 诗词生成项目

基于字符级 RNN 的七言绝句生成项目，使用 `LSTM/GRU` 做自回归语言建模，支持训练、断点续训、模型评测、首句续写、藏头诗生成、采样策略横评、消融实验与可视化分析。

项目默认围绕固定长度的七言绝句展开：每首诗清洗后保留 28 个汉字，并编码为 `[BOS] + 28字 + [EOS]` 的序列形式进行训练。

## 项目功能
- 数据清洗与缓存：从原始诗歌文本中过滤出合法七言绝句，缓存清洗结果和词表。
- 主模型训练：训练默认的双层 `LSTM` 主模型，并保存最佳与最终 checkpoint。
- 断点续训：从已有 checkpoint 恢复模型、优化器和历史日志，继续训练。
- 模型评测：计算训练集、验证集、测试集的损失与 `PPL`，并导出采样横评结果。
- 条件生成：支持首句续写、藏头诗生成，并提供温度采样、`top-k` 和韵律引导。
- 消融实验：对层数、宽度、正则化、循环单元类型等因素做控制变量对比。
- 可视化分析：生成训练曲线、学习率衰减、梯度范数、消融实验对比图和采样策略图。

## 项目结构
```text
poem generate/
├─ config.py                    全局路径、训练参数、消融实验配置
├─ data_utils.py                数据清洗、编码、词表构建、Dataset/DataLoader
├─ model.py                     字符级 LSTM/GRU 语言模型
├─ trainer.py                   单轮训练、评估、完整 fit 流程、checkpoint 保存
├─ train.py                     主模型训练入口
├─ resume_train.py              断点续训入口
├─ evaluate.py                  测试集评测与采样参数横评
├─ generate.py                  首句续写 / 藏头诗交互生成
├─ sampler.py                   temperature 与 top-k 采样
├─ metrics.py                   格式、重复度、押韵率、多样性等指标
├─ train visualization.py       主模型训练过程可视化
├─ compare visualize.py         采样参数横评可视化
├─ ablation_train.py            消融实验训练入口
├─ ablation_evaluate.py         消融实验评测
├─ ablation_visualize.py        消融实验图表生成
├─ utils.py                     随机种子、设备、JSON、参数统计等工具
├─ checkpoints/                 模型权重输出目录
├─ cache/                       清洗语料与词表缓存目录
├─ outputs/
│  ├─ figures/                  训练、横评、消融实验图表
│  ├─ metrics/                  横评结果与指标数据
│  ├─ samples/                  生成样例输出
│  ├─ ablation/                 消融实验结果与清单
│  └─ recovered_logs/           从 checkpoint 恢复出的日志 CSV
└─ README.md                    本文档
```

## 核心流程
### 1. 数据处理
- 原始数据默认读取根目录的 `chinese_poem.txt`。
- 通过 `data_utils.py` 清洗文本，仅保留纯汉字且长度恰好为 28 的七言绝句。
- 数据按 `8:1:1` 划分训练、验证、测试集。
- 词表基于训练集统计，保留特殊 token：`[PAD]`、`[BOS]`、`[EOS]`、`[UNK]`。
- 首次运行会生成 `cache/poems_clean.txt` 和 `cache/vocab.json`，后续直接复用缓存。

### 2. 模型训练
- 默认主模型配置在 `config.py` 中定义。
- 当前主模型为：
  - `model_type=lstm`
  - `embed_dim=256`
  - `hidden_dim=512`
  - `num_layers=2`
  - `dropout=0.3`
- 训练时使用：
  - `Adam`
  - `CrossEntropyLoss`
  - `ReduceLROnPlateau`
  - 梯度裁剪 `1.0`
  - 早停策略
- checkpoint 默认保存在 `checkpoints/` 下：
  - `poem_lstm_final_best.pth`
  - `poem_lstm_final_last.pth`

### 3. 生成与评测
- `generate.py` 支持两种主要条件生成：
  - 首句续写
  - 藏头诗
- 支持 `temperature`、`top-k` 和默认开启的韵律引导。
- `evaluate.py` 用于生成采样参数横评结果，比较多样性、重复度和押韵率。

### 4. 消融实验
- `ablation_train.py` 会逐个训练 `config.py` 中定义的消融实验配置。
- `ablation_evaluate.py` 对每个消融模型统一做测试集评测与样例生成。
- `ablation_visualize.py` 将结果绘制为 PPL、多样性、参数量关系等图表。

## 主要脚本说明
### 训练相关
- `train.py`：训练主模型；若 `config.py` 中 `SKIP_TRAINING=True`，则直接加载最佳模型做最终评估。
- `resume_train.py`：从 checkpoint 继续训练额外 epoch。
- `recover_logs.py`：从 checkpoint 恢复 `history`，导出 CSV 或仿照训练日志格式打印。

### 评测与生成
- `evaluate.py`：评测模型在测试集上的表现，并做采样参数横评。
- `generate.py`：交互式生成首句续写和藏头诗，支持韵律引导开关。

### 可视化
- `train visualization.py`：生成主模型训练过程图表，输出到 `outputs/figures/`。
- `compare visualize.py`：生成采样策略对比图表。
- `ablation_visualize.py`：生成消融实验对比图表。

## 快速开始
### 1. 环境准备
建议安装：
- `python`
- `torch`
- `numpy`
- `matplotlib`
- `pypinyin`（可选，用于韵律引导和押韵分析）

### 2. 训练主模型
```bash
python train.py
```

### 3. 断点续训
```bash
python resume_train.py
```

### 4. 交互生成
```bash
python generate.py
```

### 5. 评测与采样横评
```bash
python evaluate.py
python "compare visualize.py"
```

### 6. 消融实验
```bash
python ablation_train.py
python ablation_evaluate.py
python ablation_visualize.py
```

### 7. 训练可视化
```bash
python "train visualization.py"
```


## 输出目录说明
- `checkpoints/`：保存最佳模型和最终模型权重。
- `cache/`：保存清洗后的语料和词表缓存。
- `outputs/figures/`：训练曲线、采样横评图、消融实验图。
- `outputs/metrics/`：横评结果 JSON / CSV。
- `outputs/samples/`：生成样例文本。
- `outputs/ablation/`：消融实验清单与评测结果。
- `outputs/recovered_logs/`：从 checkpoint 恢复的逐 epoch 日志。

## 模型与实现要点
- 模型本质是字符级自回归语言模型。
- 输入长度固定为 29：`[BOS] + 28字`。
- 目标长度固定为 29：`28字 + [EOS]`。
- 前向输出形状为 `(B, T, V)`，训练时展平为 `(B*T, V)` 计算交叉熵。
- 固定长度七言绝句的设定简化了 padding 和 mask 处理。
- 在 Windows + CUDA 环境下，项目会禁用 cuDNN RNN 路径以避免已知退出崩溃问题。


## 使用提示
- 若想重新训练主模型，请检查 `config.py` 中的 `SKIP_TRAINING` 是否为 `False`。
- 若未安装 `pypinyin`，韵律引导和押韵分析会自动降级。
- Windows 下运行脚本时，项目已内置若干稳定性处理，无需手动修改 cuDNN 配置。

