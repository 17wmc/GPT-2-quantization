import random
import torch
from transformers import AutoTokenizer, pipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset

# 定义模型ID和输出目录
model_id = "gpt2"
out_dir = model_id + "-GPTQ"

# 加载量化定义、模型、分词器
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    damp_percent=0.01,
    desc_act=False,
)
model = AutoGPTQForCausalLM.from_pretrained(model_id, quantize_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 加载数据并对示例进行分词
n_samples = 1024


data = load_dataset("allenai/c4", data_files="en/c4-train.00001-of-01024.json.gz", split=f"train[:{n_samples * 5}]")


tokenized_data = tokenizer("\n\n".join(data['text']), return_tensors='pt')

# 格式化分词的示例
examples_ids = []
for _ in range(n_samples):
    i = random.randint(0, tokenized_data.input_ids.shape[1] - tokenizer.model_max_length - 1)
    j = i + tokenizer.model_max_length
    input_ids = tokenized_data.input_ids[:, i:j]
    attention_mask = torch.ones_like(input_ids)
    examples_ids.append({'input_ids': input_ids, 'attention_mask': attention_mask})

# 使用 GPTQ 进行量化
model.quantize(
    examples_ids,
    batch_size=1,
    use_triton=True,
)

# 保存模型和分词器
model.save_quantized(out_dir, use_safetensors=True)
tokenizer.save_pretrained(out_dir)

device = "cuda:7" if torch.cuda.is_available() else "cpu"

# 重新加载量化模型和分词器
model = AutoGPTQForCausalLM.from_quantized(
    out_dir,
    device=device,
    use_triton=True,

    use_safetensors=True,

)
tokenizer = AutoTokenizer.from_pretrained(out_dir)

# 使用模型进行文本生成
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
result = generator("The story is about:", do_sample=True, max_length=50)[0]['generated_text']
print(f"原始模型回答:\n{result}")



