from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from copy import deepcopy
torch.manual_seed(0)

# Set device to CPU for now
device = 'cpu'

# Load model and tokenizer
model_id = 'gpt2'
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Print model size
print(f"模型的大小: {model.get_memory_footprint():,} bytes")



def absmax_quantize(X):
    # 计算比例因子
    scale = 127 / torch.max(torch.abs(X))
    # 量化
    X_quant = (scale * X).round()
    # 反量化
    X_dequant = X_quant / scale
    return X_quant.to(torch.int8), X_dequant

def zeropoint_quantize(X):
    # 计算数据范围
    x_range = torch.max(X) - torch.min(X)
    x_range = 1 if x_range == 0 else x_range
    # 计算比例因子
    scale = 255 / x_range
    # 计算比例因子
    zeropoint = (-scale * torch.min(X) - 128).round()
    # 量化
    X_quant = torch.clip((X * scale + zeropoint).round(), -128, 127)
    # 反量化
    X_dequant = (X_quant - zeropoint) / scale
    return X_quant.to(torch.int8), X_dequant

# 提取第一个注意力层的权重
weights = model.transformer.h[0].attn.c_attn.weight.data
print("原始权重:")
print(weights)

# 用 absmax 方法量化
weights_abs_quant, _ = absmax_quantize(weights)
print("\nAbsmax量化权重:")
print(weights_abs_quant)

# 用 zeropoint 方法量化
weights_zp_quant, _ = zeropoint_quantize(weights)
print("\nZero-point量化权重:")
print(weights_zp_quant)



# 保存原始权重
weights = [param.data.clone() for param in model.parameters()]

# 创建量化模型副本
model_abs = deepcopy(model)

# 量化所有模型权重
weights_abs = []
for param in model_abs.parameters():
    _, dequantized = absmax_quantize(param.data)
    param.data = dequantized
    weights_abs.append(dequantized)

# 创建量化模型副本
model_zp = deepcopy(model)

# 量化所有模型权重
weights_zp = []
for param in model_zp.parameters():
    _, dequantized = zeropoint_quantize(param.data)
    param.data = dequantized
    weights_zp.append(dequantized)

def generate_text(model, input_text, max_length=50):
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    output = model.generate(inputs=input_ids,
                            max_length=max_length,
                            do_sample=True,
                            top_k=30,
                            pad_token_id=tokenizer.eos_token_id,
                            attention_mask=input_ids.new_ones(input_ids.shape))
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 原始模型和量化模型的生成句子
original_text = generate_text(model, "The story is about:")
absmax_text   = generate_text(model_abs, "The story is about:")
zp_text       = generate_text(model_zp, "The story is about:")

print(f"原始模型回答:\n{original_text}")
print("-" * 50)
print(f"Absmax量化模型回答:\n{absmax_text}")
print("-" * 50)
print(f"Zeropoint量化模型回答:\n{zp_text}")

def calculate_perplexity(model, text):
    # 分词器编码
    encodings = tokenizer(text, return_tensors='pt').to(device)

    # 定义输入和对比
    input_ids = encodings.input_ids
    target_ids = input_ids.clone()

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

    # 计算loss
    neg_log_likelihood = outputs.loss

    # 困惑度计算
    ppl = torch.exp(neg_log_likelihood)

    return ppl

ppl     = calculate_perplexity(model, original_text)
ppl_abs = calculate_perplexity(model_abs, absmax_text)
ppl_zp  = calculate_perplexity(model_zp, absmax_text)

print(f"原始模型困惑度:  {ppl.item():.2f}")
print(f"Absmax量化模型回答困惑度:    {ppl_abs.item():.2f}")
print(f"Zeropoint量化模型回答困惑度: {ppl_zp.item():.2f}")
