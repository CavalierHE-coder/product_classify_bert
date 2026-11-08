import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from config.config import *
from process.dataset import get_dataset
from runner.train import Trainer

def evaluate():
    # 提示训练开始
    print("开始评估...")
    # 1. 定义设备
    # 有 GPU 用 GPU，否则用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 2. 加载分词器
    # 按模型名加载分词器（DataCollatorWithPadding 要靠它获取 pad 信息）
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # 3. 加载模型
    # 打开预处理阶段保存的标签文件
    with open(MODEL_DIR/LABELS_FILE, "r", encoding="utf-8") as f:
        # 按行读成列表，行号就是类别编号
        all_labels = f.read().splitlines()
    # 加载"预训练 BERT + 随机初始化的分类头"
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR/'best').to(device)
    # 4. 定义训练器
    # 获取测试集dataset
    test_dataset = get_dataset('test')
    # 定义数据整理器
    collator = DataCollatorWithPadding(
        # 用分词器提供 pad 的 token id
        tokenizer=tokenizer,
        # 按 batch 内最长样本动态补齐
        padding=True,
    )
    # 定义计算评估指标的函数
    # 输入预测列表和真实标签列表，返回指标字典
    def compute_metrics(predictions, labels) -> dict:
        # 准确率：预测正确的比例
        accuracy = accuracy_score(labels, predictions)
        # 精确率：weighted 表示按各类别样本数加权平均
        precision = precision_score(labels, predictions, average='weighted')
        # 召回率：同样加权平均
        recall = recall_score(labels, predictions, average='weighted')
        # F1：精确率和召回率的调和平均
        f1 = f1_score(labels, predictions, average='weighted')
        # 返回字典
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    # 定义训练器
    trainer = Trainer(
        # 模型
        model=model,
        # 训练集
        train_dataset=None,
        # 验证集
        valid_dataset=test_dataset,
        # 数据整理器
        collator=collator,
        # 指标计算函数
        compute_metrics=compute_metrics,
        # 设备
        device=device,
    )
    # 5. 评估
    metrics = trainer.evaluate()
    print(f"评估结果：{metrics}")

if __name__ == '__main__':
    evaluate()