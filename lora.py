# -*- coding: utf-8 -*-
# @Time:2024/7/15 11:17
# @File:lora.py
# @software:PyCharm
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer


class LoRAAdapter(nn.Module):
    def __init__(self, original_weight, rank=4):
        super(LoRAAdapter, self).__init__()
        self.rank = rank
        self.A = nn.Parameter(torch.randn(original_weight.size(0), rank))
        self.B = nn.Parameter(torch.randn(rank, original_weight.size(1)))

    def forward(self, W0):
        return W0 + torch.matmul(self.A, self.B)


# 加载预训练模型
model_name_or_path = "GPT2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2Model.from_pretrained(model_name_or_path)

# 在 PyTorch 中使用这个模型获取给定文本的特征的方法如下：
text = "今天天气非常好"
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print(output)
#
# 获取模型的权重参数
for name, param in model.named_parameters():
    print(f"参数名称: {name}, 形状: {param.shape}")
#
# 获取原始模型的权重矩阵
original_weight = None
for name, param in model.named_parameters():
    if 'attn.c_attn.weight' in name:
        original_weight = param
        break

if original_weight is None:
    raise ValueError("在模型中找不到注意力权重。")
#
# 初始化LoRA适配器
lora_adapter = LoRAAdapter(original_weight)

# 定义优化器，只优化LoRA的参数
optimizer = torch.optim.Adam(lora_adapter.parameters(), lr=1e-4)

tokenizer.pad_token = tokenizer.eos_token


# 数据加载器 (假设已经有一个数据集)
def get_dataloader():
    # 这里使用一个简单的示例数据集
    texts = ["你好，你好吗？", "我很好，谢谢！", "你叫什么名字？"]
    encodings = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    dataset = torch.utils.data.TensorDataset(encodings['input_ids'])
    return torch.utils.data.DataLoader(dataset, batch_size=2)


dataloader = get_dataloader()


# 定义训练过程
def train(model, lora_adapter, dataloader, optimizer, epochs=3):
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            inputs = batch[0]
            outputs = model(input_ids=inputs).last_hidden_state

            # 简单的损失函数 (示例)
            loss = outputs.mean()  # 通常你会有一个更复杂的损失函数

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新模型权重
            with torch.no_grad():
                updated_weight = lora_adapter(original_weight)
                for layer in model.h:
                    layer.attn.c_attn.weight.copy_(updated_weight)

        print(f"第 {epoch + 1}/{epochs} 轮，损失: {loss.item()}")


# 执行微调
train(model, lora_adapter, dataloader, optimizer)
