from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from copy import deepcopy
from absmax-zeropoint import generate_text, calculate_perplexity
torch.manual_seed(0)


device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

model_int8 = AutoModelForCausalLM.from_pretrained(model_id,
                                             device_map='auto',
                                             load_in_8bit=True,
                                             )
print(f"模型大小: {model_int8.get_memory_footprint():,} bytes")

# Generate text with quantized model
text_int8 = generate_text(model_int8, "The story is about:")

print(f"原始模型回答:\n{original_text}")
print("-" * 50)
print(f"LLM.int8()量化模型回答:\n{text_int8}")
print(f"原始模型回答困惑度:   {ppl.item():.2f}")

ppl = calculate_perplexity(model_int8, text_int8)
print(f"LLM.int8()量化模型回答困惑度: {ppl.item():.2f}")

