# 基于 BERT 的商品分类

[English](README.md) | [简体中文](README.zh-CN.md)

通过微调 `bert-base-chinese`,把中文电商商品标题分到 30 个类别中。项目覆盖完整流程:数据预处理、带早停的训练、评估、预测以及 Web 演示。

## 特性

- **完整流程**:预处理、训练、评估、推理和 Web 演示
- **手写训练循环**:轻量的 `Trainer`,支持定期验证、保存最优模型和早停
- **动态补齐**:使用 `DataCollatorWithPadding` 按 batch 补齐序列,节省显存和计算
- **丰富的评估指标**:loss、accuracy,以及加权的 precision / recall / F1
- **TensorBoard 日志**:训练曲线写入 `logs/` 下按时间命名的目录

## 项目结构

```
product_classify_bert/
├── checkpoints/          # 微调后的模型和标签列表
├── data/                 # 原始数据(TSV)和处理后的数据集
├── logs/                 # TensorBoard 日志
├── pretrained/           # 预训练模型文件
└── src/
    ├── config/           # 路径和超参数配置
    ├── process/          # 数据预处理与数据集加载
    │   ├── preprocess.py
    │   └── dataset.py
    ├── runner/           # 训练、评估、预测
    │   ├── train.py
    │   ├── evaluate.py
    │   └── predict.py
    ├── web/              # Web 演示
    └── main.py           # 入口文件
```

## 数据集

带类别标签的商品标题,使用制表符分隔的文件存储:

| 划分 | 样本数 |
|---|---|
| 训练集 | 88,824 |
| 验证集 | 22,207 |
| 测试集 | 22,207 |

字段说明:

| 字段 | 说明 |
|---|---|
| `text_a` | 商品标题 |
| `label` | 类别名称 |

30 个类别:3C数码、个护、书籍、乳品、休闲食品、健康、健康食品、办公、宠物、家居、家电、服饰内衣、母婴、水产、水果、汽车用品、清洁、玩具、礼品、粮油速食、美妆、肉禽蛋、蔬菜、运动、酒饮冲调、钟表配饰、鞋靴箱包、餐饮、香烟、鲜花绿植。

## 安装

```bash
# 创建环境
conda create -n NLP python=3.12
conda activate NLP

# 安装依赖
pip install torch transformers datasets scikit-learn tensorboard tqdm
```

在 `src/config/config.py` 中配置模型名称或路径(`MODEL_NAME`),以及数据和输出路径。

## 使用方法

代码里会导入 `config.config` 这样的模块,所以要保证 `src` 在 Python 路径中(例如在 PyCharm 中把 `src` 标记为 *Sources Root*,或者设置 `PYTHONPATH`)。

```bash
# Linux / macOS
export PYTHONPATH=src
# Windows (cmd)
set PYTHONPATH=src
```

### 1. 数据预处理

```bash
python src/process/preprocess.py
```

这一步会:

1. 加载训练集、验证集、测试集的 TSV 文件
2. 从训练集提取排序后的标签列表,并保存为标签文件
3. 把字符串标签映射成整数编号,并转换成 `ClassLabel` 类型
4. 对标题分词(只截断、不补齐),保留 `input_ids`、`token_type_ids`、`attention_mask`、`labels`
5. 把处理后的数据集保存到磁盘

### 2. 训练

```bash
python src/runner/train.py
```

训练细节:

- 模型:`AutoModelForSequenceClassification`,输出 30 个类别
- 优化器:Adam
- 每 `SAVE_STEPS` 步:记录训练 loss、在验证集上评估,并写入 TensorBoard
- 监控指标(默认是验证集 loss)有提升时保存模型
- 连续 `EARLY_STOP_PATIENCE` 次验证没有提升则提前停止训练

### 3. 评估

```bash
python src/runner/evaluate.py
```

### 4. 预测

```bash
python src/runner/predict.py
```

### 5. Web 演示

```bash
python src/main.py
```

请根据你实际的 `main.py` 和 `src/web` 使用的框架调整上面的命令。

### 6. 用 TensorBoard 查看训练过程

```bash
tensorboard --logdir logs
```

## 配置说明

`src/config/config.py` 中的主要配置:

| 名称 | 说明 |
|---|---|
| `MODEL_NAME` | 预训练模型名称或本地路径 |
| `EPOCHS` | 训练轮数 |
| `BATCH_SIZE` | 批大小 |
| `LEARNING_RATE` | 学习率(BERT 微调常用 2e-5 到 5e-5) |
| `SAVE_STEPS` | 每 N 步验证并记录一次 |
| `EARLY_STOP_PATIENCE` | 早停耐心值,即连续多少次验证没有提升就停止 |
| `RAW_DATA_DIR` / `PROCESSED_DATA_DIR` | 原始数据和处理后数据的目录 |
| `MODEL_DIR` / `LOGS_DIR` | 模型输出和日志目录 |

## 实验结果

`src/runner/evaluate.py` 在测试集上的评估结果:

| 指标 | 数值 |
|---|---|
| Loss | 0.2382 |
| Accuracy | 93.34% |
| Precision(加权) | 93.36% |
| Recall(加权) | 93.34% |
| F1(加权) | 93.32% |

## 注意事项

- 预处理、训练和推理要使用同一个分词器和同一个模型。
- 标签文件决定了编号和类别名的对应关系,训练完成后不要修改。
- 建议使用 GPU 训练;没有 CUDA 时脚本会自动回退到 CPU。

## 许可证

MIT
