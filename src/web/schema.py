from pydantic import BaseModel

# 定义商品类
class Item(BaseModel):
    name: str

# 定义返回结果类
class Result(BaseModel):
    category: str