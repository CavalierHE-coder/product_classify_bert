import torch
import uvicorn
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from runner.predict import Predictor
from web.schema import Item,Result
from config.config import *

# 创建app对象
app = FastAPI()

# 1. 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 2. 加载分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# 3. 加载模型
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR / 'best')
# 4. 定义预测器
predictor = Predictor(model=model, tokenizer=tokenizer, device=device)

@app.post("/predict")
def predict(item: Item):
    category = predictor.predict(item.name)
    return Result(category=category)

def run_app():
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == '__main__':
    run_app()