import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config.config import *

# 自定义预测器
class Predictor:
    def __init__(self, model, tokenizer, device):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    # 核心方法，传入文本，返回预测分类(中文)
    def predict(self, text: str | list[str]):
        # 判断是否是单个字符串，统一为列表形式
        is_str = isinstance(text, str)
        if is_str:
            text = [text]

        # 1. 编码，得到模型输入
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)

        # 2. 前向传播
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 3. 根据输出值，得到预测标签
        pred_ids = torch.argmax(outputs.logits, dim=-1).tolist()

        # 4. 转换成label名称
        pred_labels = [ self.model.config.id2label[id] for id in pred_ids ]

        # 5. 判断如果是单条数据，直接返回元素
        if is_str:
            return pred_labels[0]
        return pred_labels

# 推理预测主流程
def predict():
    # 1. 定义设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 2. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # 3. 加载模型
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR/'best')
    # 4. 创建预测器
    predictor = Predictor(model, tokenizer, device)
    # 5. 调用预测方法
    text = '好奇心钻装纸尿裤L40片9-14kg'
    print(predictor.predict(text))

    texts = ['911-267遥控车', '瓦伦丁Wurenbacher小麦西柚啤酒500ml*12听整箱装德国原装进口果啤', '640G正航牛奶早餐饼干']
    print(predictor.predict(texts))

if __name__ == '__main__':
    predict()