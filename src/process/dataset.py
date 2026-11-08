from datasets import load_from_disk
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from config.config import *

# ========== 1. 加载数据集 ==========
# 参数 ds_type 取 'train' / 'valid' / 'test'，默认 'train'
def get_dataset(ds_type='train'):
    # 拼出子目录路径：preprocess 里 save_to_disk 会按划分名建子目录
    path = str( PROCESSED_DATA_DIR / ds_type )
    # 从磁盘读取该划分，得到一个 Dataset 对象（内存映射读取，不会一次性占满内存）
    dataset = load_from_disk(path)
    # 返回数据集
    return dataset

# ========== 2. 构建 DataLoader ==========
# tokenizer 用来做动态补齐，ds_type 指定用哪个划分
def get_dataloader(tokenizer, ds_type):
    # 先加载对应划分的数据集
    dataset = get_dataset(ds_type)
    # 设置 Tensor 格式：取数据时把各列自动转成 torch.Tensor，而不是 Python 列表
    dataset.set_format(type='torch')
    # 定义数据整理器：负责把一个 batch 里长短不一的样本拼成统一形状
    collator = DataCollatorWithPadding(
        # 用分词器获取 pad 的 token id、补齐方向等信息
        tokenizer=tokenizer,
        # padding=True：补齐到当前 batch 里最长样本的长度（动态补齐）
        padding=True,
    )
    # 定义加载器：每次取 BATCH_SIZE 条，打乱顺序，并用 collator 拼接成 batch
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
    # 返回加载器
    return dataloader

# ========== 3. 测试入口 ==========
# 只有直接运行本文件时才执行，被别的文件 import 时不会执行
if __name__ == '__main__':
    # 按模型名加载分词器（和预处理时必须是同一个模型）
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # 获取训练集的 DataLoader
    dataloader = get_dataloader(tokenizer, ds_type='train')
    # 遍历 DataLoader，每次得到一个 batch（一个字典）
    for batch in dataloader:
        # 遍历 batch 里的每个字段：k 是字段名，v 是对应的张量
        for k, v in batch.items():
            # 打印字段名和张量形状
            print(k, "→", v.shape)
            # 打印张量内容
            print(v)
        # 只看第一个 batch，验证数据格式没问题就退出
        break