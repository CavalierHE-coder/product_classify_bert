import time
from dataclasses import dataclass
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config.config import *
from process.dataset import *
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score



# ========== 1. 训练配置类 ==========
@dataclass
# 定义训练配置类：把所有超参数集中放在一起
class TrainConfig:
    # 训练轮数，默认取 config 里的 EPOCHS
    epochs: int = EPOCHS
    # 每个 batch 的样本数
    batch_size: int = BATCH_SIZE
    # 学习率
    learning_rate: float = LEARNING_RATE
    # 每隔多少步做一次"日志 + 验证 + 早停判断"
    save_steps: int = SAVE_STEPS  # 50
    # 模型保存目录
    output_dir: str = MODEL_DIR
    # TensorBoard 日志目录
    log_dir: str = LOGS_DIR
    # 早停耐心值：连续多少次验证没有提升就停止
    early_stop_patience: int = EARLY_STOP_PATIENCE  # 5
    # 用哪个指标判断是否提升，默认看验证集 loss
    early_stop_metric: str = 'loss'
    # 定义是否开启AMP功能
    use_amp: bool = True



# ========== 2. 训练器类 ==========
# 定义训练器工具类：把训练、验证、早停、保存封装在一起
class Trainer:
    # 初始化，参数：模型、训练集、验证集、数据整理器、指标计算函数、设备、配置
    def __init__(self, model, train_dataset, valid_dataset, collator,
                 compute_metrics, device, config=TrainConfig()):
        # 把模型搬到指定设备（GPU/CPU）；对 nn.Module 来说 .to 是原地操作
        self.model = model.to(device)
        # 保存训练集
        self.train_dataset = train_dataset
        # 保存验证集
        self.valid_dataset = valid_dataset
        # 保存数据整理器（负责动态补齐）
        self.collator = collator
        # 保存外部传入的指标计算函数
        self.compute_metrics = compute_metrics
        # 保存设备
        self.device = device
        # 保存配置
        self.config = config
        # 创建 Adam 优化器，管理模型全部参数
        self.optimizer = Adam(model.parameters(), lr=config.learning_rate)
        # Tensorboard写入器，日志目录 = 日志根目录/当前时间，每次运行单独一个文件夹，方便对比
        self.writer = SummaryWriter( str(Path(config.log_dir) / time.strftime("%Y-%m-%d_%H-%M-%S")))
        # 全局步数计数器，跨 epoch 累计
        self.step = 0
        # 历史最高评分，初始为负无穷，保证第一次验证一定算"提升"
        self.early_stop_best_score = -float('inf')
        # 连续没提升的次数
        self.early_stop_counter = 0
        # AMP梯度缩放器
        self.scaler = torch.amp.GradScaler(device=device, enabled=config.use_amp)
        # 检查点路径
        self.checkpoint_path = Path(self.config.output_dir) / 'last' / 'checkpoint.pt'

    # 定义内部方法：获取数据加载器
    def _get_dataloader(self, dataset):
        # 设置Tensor格式：取数据时自动转成 torch.Tensor
        dataset.set_format(type='torch')
        # 定义加载器
        dataloader = DataLoader(
            # 要加载的数据集
            dataset,
            # 每批样本数
            batch_size=self.config.batch_size,
            # 每个 epoch 打乱顺序
            shuffle=True,
            # 用 collator 把一批样本补齐并拼接
            collate_fn=self.collator,
        )
        # 返回加载器
        return dataloader

    # 定义内部函数：训练一个批次(一步迭代)
    def _train_one_step(self, inputs):
        # 切到训练模式：启用 dropout 等
        self.model.train()
        # 把这批数据的每个张量搬到模型所在设备
        inputs = { k:v.to(self.device) for k,v in inputs.items() }
        # 前向传播
        with torch.autocast(device_type=self.device.type,
                            dtype=torch.float16,
                            enabled=self.config.use_amp  # 开启AMP功能
                            ):
            # 传入 input_ids 等字段；因为含 labels，模型会自动算出 loss
            outputs = self.model(**inputs)
        # 反向传播，计算每个参数的梯度
        self.scaler.scale(outputs.loss).backward()
        # 更新参数，优化器根据梯度更新参数
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # 清空梯度，防止累积到下一步
        self.optimizer.zero_grad()
        # 返回 Python 数字形式的损失，用于日志
        return outputs.loss.item()

    # 定义内部函数：判断是否早停，以及保存模型
    def _should_early_stop(self, metrics):
        # 取出判定早停的指标
        metric = metrics[self.config.early_stop_metric]
        # 转换成评分
        # loss 越小越好，取负数后统一成"分数越大越好"；accuracy、f1 等本身越大越好
        score = -metric if self.config.early_stop_metric == 'loss' else metric
        # 如果评分超过历史最高，就保存模型
        if score > self.early_stop_best_score:
            # 更新历史最高分
            self.early_stop_best_score = score
            # 计数器清零
            self.early_stop_counter = 0  # 计数器清零
            tqdm.write("保存最佳模型...")
            # 保存当前最优模型（权重 + 配置）
            self.model.save_pretrained(Path(self.config.output_dir)/'best')
            # 不早停
            return False
        # 没有提升
        else:
            # 如果评分降低，计数加1
            self.early_stop_counter += 1
            # 如果超过容忍度，就早停
            if self.early_stop_counter >= self.config.early_stop_patience:
                # 连续没提升的次数达到耐心值，早停
                return True
            # 还没到耐心值
            else:
                # 继续训练
                return False

    # 定义内部函数：保存检查点
    def _save_checkpoint(self):
        # 定义检查点，字典
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'step': self.step,
            'early_stop_best_score': self.early_stop_best_score,
            'early_stop_counter': self.early_stop_counter,
        }
        # 保存文件
        tqdm.write("保存检查点...")
        torch.save(checkpoint, self.checkpoint_path)

    # 定义内部函数：加载检查点
    def _load_checkpoint(self):
        # 判断检查点是否存在
        if self.checkpoint_path.exists():
            print("发现检查点，继续训练...")
            # 获取检查点
            checkpoint = torch.load(self.checkpoint_path)
            # 加载所有状态
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scaler.load_state_dict(checkpoint['scaler'])
            self.step = checkpoint['step']
            self.early_stop_best_score = checkpoint['early_stop_best_score']
            self.early_stop_counter = checkpoint['early_stop_counter']
        else:
            print("没有发现检查点，从头开始训练...")

    # 核心方法
    # 训练主流程
    def train(self):
        # 加载检查点
        self._load_checkpoint()
        # 获取数据加载器
        dataloader = self._get_dataloader(self.train_dataset)
        # 循环迭代
        # 外层：按轮数循环
        for epoch in range(self.config.epochs):
            # 内层：遍历每个 batch，tqdm 显示进度条
            for inputs in tqdm(dataloader, desc=f"[训练：Epoch {epoch + 1}]"):
                # 调用一批数据的训练过程，得到损失
                this_loss = self._train_one_step(inputs)
                # 全局步数加 1
                self.step += 1
                # 判断是否达到50次迭代，做验证，进行日志和模型保存
                # 步数是 save_steps 的整数倍时，才做日志、验证和早停判断
                if self.step % self.config.save_steps == 0:
                    # 保存训练日志
                    # 用 tqdm.write 打印，避免打乱进度条
                    tqdm.write(f'[Epoch: {epoch+1} | Step: {self.step}] Loss: {this_loss:.4f}')
                    # 把训练 loss 写入 TensorBoard，横轴是全局步数
                    self.writer.add_scalar('loss', this_loss, self.step)
                    # 进行验证，得到评估指标
                    metrics = self.evaluate()
                    # 把指标字典拼成 "loss:0.1234|accuracy:0.9500|..." 的字符串
                    metrics_str = '|'.join([f'{k}:{v:.4f}' for k,v in metrics.items()])
                    # 打印验证指标
                    tqdm.write(f'[Evaluate] {metrics_str}')

                    # 早停逻辑
                    # 判断是否早停（有提升时内部会顺便保存模型）
                    if self._should_early_stop(metrics):
                        # 提示早停
                        tqdm.write("早停！")
                        # 退出整个函数，两层循环一起结束
                        return  # 退出整个函数

                    # 定期保存检查点
                    self._save_checkpoint()

    # 验证评估方法，返回不同指标(字典)
    def evaluate(self) -> dict:
        # 获取验证集的数据加载器
        dataloader = self._get_dataloader(self.valid_dataset)
        # 切到评估模式：关闭 dropout，结果稳定
        self.model.eval()
        # 记录总损失，用于计算指标1 - 验证集平均损失
        total_loss = 0.0
        # 记录所有数据的预测标签和真实标签，用于计算其它指标(准确率、精确率、f1...)
        # 预测标签列表
        predictions = []
        # 真实标签列表
        labels = []
        # 遍历验证集
        # 不记录梯度：省显存、加速
        with torch.no_grad():
            # 遍历验证集的每个 batch
            for inputs in tqdm(dataloader, desc=f"[验证]"):
                # 数据搬到设备上
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                # 前向传播
                outputs = self.model(**inputs)
                # 获取损失
                # 这一批的平均损失（Python 数字）
                loss = outputs.loss.item()
                # 累加到总损失
                total_loss += loss
                # 添加预测结果到列表
                # logits 形状 [batch, 类别数]，取最大值所在下标作为预测类别
                this_prediction = torch.argmax(outputs.logits, dim=-1)
                # 转成 Python 列表后追加
                predictions.extend(this_prediction.tolist())
                # 添加真实标签到列表
                # 取出这一批的真实标签
                this_labels = inputs['labels']
                # 转成 Python 列表后追加
                labels.extend(this_labels.tolist())
        # 计算评估指标
        # 指标1：平均损失
        # 总损失除以 batch 数，得到验证集平均损失
        loss = total_loss / len(dataloader)
        # 指标2：其他指标，调用外部传入的计算函数，返回字典
        metrics = self.compute_metrics(predictions, labels)
        # 合并成一个字典返回：loss 在前，后面是准确率等指标
        return {'loss': loss, **metrics}

# ========== 3. 训练入口函数 ==========
# 组装所有组件并启动训练
def train():
    # 提示训练开始
    print("开始训练...")
    # 1. 定义设备，有 GPU 用 GPU，否则用 CPU
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
    model = AutoModelForSequenceClassification.from_pretrained(
        # 预训练模型名
        MODEL_NAME,
        # 分类头输出维度 = 类别数（30）
        num_labels=len(all_labels),
        # 编号 -> 类别名，写进模型配置，推理时可直接得到类别名
        id2label={ id:label for id, label in enumerate(all_labels)},
        # 类别名 -> 编号
        label2id={ label:id for id, label in enumerate(all_labels)}
    # 搬到设备上
    ).to(device)
    # 4. 定义训练器
    # 获取训练集dataset
    train_dataset = get_dataset('train')
    # 获取验证集 dataset
    valid_dataset = get_dataset('valid')
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
        train_dataset=train_dataset,
        # 验证集
        valid_dataset=valid_dataset,
        # 数据整理器
        collator=collator,
        # 指标计算函数
        compute_metrics=compute_metrics,
        # 设备
        device=device,
        # 训练配置，全部用默认值
        config=TrainConfig()
    )
    # 5. 训练
    trainer.train()

# 直接运行本文件时才执行，被 import 时不会执行
if __name__ == "__main__":
    # 调用训练入口
    train()