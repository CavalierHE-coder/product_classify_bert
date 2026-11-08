# Product Classification with BERT

[English](README.md) | [简体中文](README.zh-CN.md)

Classify Chinese e-commerce product titles into 30 categories by fine-tuning `bert-base-chinese`. The project covers the full pipeline: data preprocessing, training with early stopping, evaluation, prediction, and a web demo.

## Features

- **End-to-end pipeline**: preprocessing, training, evaluation, inference, and a web demo
- **Hand-written training loop**: a lightweight `Trainer` with periodic validation, best-model saving, and early stopping
- **Dynamic padding**: sequences are padded per batch with `DataCollatorWithPadding`, which saves memory and compute
- **Rich metrics**: loss, accuracy, and weighted precision / recall / F1
- **TensorBoard logging**: training curves are written to a timestamped directory under `logs/`

## Project Structure

```
product_classify_bert/
├── checkpoints/          # Fine-tuned model and label list
├── data/                 # Raw data (TSV) and processed datasets
├── logs/                 # TensorBoard logs
├── pretrained/           # Pretrained model files
└── src/
    ├── config/           # Paths and hyperparameters
    ├── process/          # Data preprocessing and dataset loading
    │   ├── preprocess.py
    │   └── dataset.py
    ├── runner/           # Training, evaluation, prediction
    │   ├── train.py
    │   ├── evaluate.py
    │   └── predict.py
    ├── web/              # Web demo
    └── main.py           # Entry point
```

## Dataset

Product titles with category labels, stored as tab-separated files:

| Split | Samples |
|---|---|
| Train | 88,824 |
| Valid | 22,207 |
| Test | 22,207 |

Columns:

| Column | Description |
|---|---|
| `text_a` | Product title |
| `label` | Category name |

The 30 categories: 3C数码, 个护, 书籍, 乳品, 休闲食品, 健康, 健康食品, 办公, 宠物, 家居, 家电, 服饰内衣, 母婴, 水产, 水果, 汽车用品, 清洁, 玩具, 礼品, 粮油速食, 美妆, 肉禽蛋, 蔬菜, 运动, 酒饮冲调, 钟表配饰, 鞋靴箱包, 餐饮, 香烟, 鲜花绿植.

## Installation

```bash
# Create the environment
conda create -n NLP python=3.12
conda activate NLP

# Install dependencies
pip install torch transformers datasets scikit-learn tensorboard tqdm
```

Set the model name or path (`MODEL_NAME`) and the data and output paths in `src/config/config.py`.

## Usage

The code imports modules such as `config.config`, so make sure `src` is on the Python path (for example, mark `src` as *Sources Root* in PyCharm, or set `PYTHONPATH`).

```bash
# Linux / macOS
export PYTHONPATH=src
# Windows (cmd)
set PYTHONPATH=src
```

### 1. Preprocess

```bash
python src/process/preprocess.py
```

This will:

1. Load the train / valid / test TSV files
2. Collect the sorted label list from the training set and save it to the label file
3. Map string labels to integer ids and cast the column to `ClassLabel`
4. Tokenize titles (`truncation=True`, no padding) and keep `input_ids`, `token_type_ids`, `attention_mask`, `labels`
5. Save the processed dataset to disk

### 2. Train

```bash
python src/runner/train.py
```

Training details:

- Model: `AutoModelForSequenceClassification` with 30 output labels
- Optimizer: Adam
- Every `SAVE_STEPS` steps: log the training loss, evaluate on the validation set, and write to TensorBoard
- The model is saved whenever the monitored metric (validation loss by default) improves
- Training stops early after `EARLY_STOP_PATIENCE` validations without improvement

### 3. Evaluate

```bash
python src/runner/evaluate.py
```

### 4. Predict

```bash
python src/runner/predict.py
```

### 5. Web demo

```bash
python src/main.py
```

Adjust the command above to match your `main.py` and the framework used in `src/web`.

### 6. Monitor with TensorBoard

```bash
tensorboard --logdir logs
```

## Configuration

Main settings in `src/config/config.py`:

| Name | Description |
|---|---|
| `MODEL_NAME` | Pretrained model name or local path |
| `EPOCHS` | Number of training epochs |
| `BATCH_SIZE` | Batch size |
| `LEARNING_RATE` | Learning rate (2e-5 to 5e-5 is typical for BERT fine-tuning) |
| `SAVE_STEPS` | Validate and log every N steps |
| `EARLY_STOP_PATIENCE` | Validations without improvement before stopping |
| `RAW_DATA_DIR` / `PROCESSED_DATA_DIR` | Raw and processed data directories |
| `MODEL_DIR` / `LOGS_DIR` | Model output and log directories |

## Results

Results from `src/runner/evaluate.py` on the test set:

| Metric | Value |
|---|---|
| Loss | 0.2382 |
| Accuracy | 93.34% |
| Precision (weighted) | 93.36% |
| Recall (weighted) | 93.34% |
| F1 (weighted) | 93.32% |

## Notes

- Keep the tokenizer and the model identical between preprocessing, training, and inference.
- The label file defines the mapping between ids and category names. Do not edit it after training.
- A GPU is strongly recommended; the script falls back to CPU automatically when CUDA is unavailable.

## License

MIT
