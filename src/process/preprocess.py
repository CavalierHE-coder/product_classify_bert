from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer
from config.config import *

# 预处理主函数：原始 CSV -> 编码好的数据集 -> 保存到磁盘
def preprocess():
    # 打印提示，标记预处理开始
    print("数据预处理开始...")
    # ========== 1. 从文件加载数据集 ==========
    # 用 csv 加载器读取三个文件，data_files 的键就是划分名，返回 DatasetDict
    dataset_dict = load_dataset('csv', data_files={
        # 训练集文件路径（Path 拼接后转成字符串）
        'train': str(RAW_DATA_DIR/TRAIN_DATA_FILE),
        # 验证集文件路径
        'valid': str(RAW_DATA_DIR/VALID_DATA_FILE),
        # 测试集文件路径
        'test': str(RAW_DATA_DIR/TEST_DATA_FILE)
    # 文件是制表符分隔的（TSV），所以指定 delimiter='\t'
    }, delimiter='\t')
    print(dataset_dict)

    # ========== 2. 加载分词器 ==========
    # 按模型名加载分词器（要和后面训练用的模型一致）
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ========== 3. 编码标签，得到类别编码 ==========
    # 3.1 获取所有标签列表
    # 取训练集 label 列去重后排序，保证每次运行编号顺序固定
    all_labels = sorted(set(dataset_dict['train']['label']))
    # 打印标签列表，确认共 30 个类别
    print(all_labels)
    # 3.2 将标签列表保存成文件
    # 以 utf-8 写入标签文件，推理时靠它把编号还原成类别名（行号 = 编号）
    with open( MODEL_DIR/LABELS_FILE, 'w', encoding='utf-8' ) as f:
        # 每个类别名占一行
        f.write('\n'.join(all_labels))
    # 3.3 将label列转成ClassLabel
    # 创建 ClassLabel 特征：names 的顺序就是编号 0~29
    class_label = ClassLabel(names=all_labels)

    # 先手动把字符串标签映射成整数
    # 构造"类别名 -> 编号"的字典，如 {'3C数码': 0, '个护': 1, ...}
    label2id = {name: i for i, name in enumerate(all_labels)}
    # 对三个划分统一做 map，把 label 列从字符串换成整数
    dataset_dict = dataset_dict.map(
        # batched=True 时 batch['label'] 是列表，用列表推导逐个查表，返回新的 label 列
        lambda batch: {'label': [label2id[x] for x in batch['label']]},
        # 批量处理，比逐条处理快
        batched=True,
    )
    # 此时 label 已是整数，再转 ClassLabel 就不会再触发字符串解析
    # 把整数列转成 ClassLabel 类型，让数据集记住编号与类别名的对应关系
    dataset_dict = dataset_dict.cast_column('label', class_label)

    # ========== 4. 编码文本（商品标题），得到模型输入 ==========
    # 定义批量编码函数，map 会把一批样本以"列名 -> 列表"的字典形式传进来
    def batch_encode(batch):
        # 对一批商品标题分词；truncation=True 超过模型最大长度（512）就截断，不补齐
        inputs = tokenizer(batch['text_a'], truncation=True)
        # 把标签放进结果里，字段名用 labels（Hugging Face 模型约定的名字）
        inputs['labels'] = batch['label']
        # 返回的字典里的键会成为新增的列
        return inputs
    # 对所有划分批量编码；remove_columns 删掉原始文本列和旧的 label 列，只保留模型需要的字段
    dataset_dict = dataset_dict.map(batch_encode, batched=True, remove_columns=['text_a', 'label'])
    # 打印训练集前 3 条，检查编码结果
    print(dataset_dict['train'][:3])

    # ========== 5. 保存数据集 ==========
    # 保存到磁盘，会按划分名生成 train/valid/test 三个子目录，之后用 load_from_disk 读取
    dataset_dict.save_to_disk(PROCESSED_DATA_DIR)

# 只有直接运行本文件时才执行，被 import 时不会执行
if __name__ == "__main__":
    # 调用预处理函数
    preprocess()