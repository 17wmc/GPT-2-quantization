# 大语言模型量化原理和GPT-2量化实战


作者：西电-极创工作室-王孟超

　　大型语言模型 (LLM) 以其广泛的计算要求而闻名。 通常，模型的大小是通过将参数（权重）数量乘以这些值的精度（数据类型）来计算的。 然而，为了节省内存，可以通过称为量化的过程使用较低精度的数据类型来存储权重。

　　在模型量化中主要的权重量化技术有**训练后量化 (PTQ：Post-Training Quantization)**和**量化感知训练（QAT：Quantization-Aware Training）**。**PTQ**是一种简单的技术，其中已训练模型的权重将转换为较低的精度，而无需任何重新训练。 尽管易于实施，但 PTQ 会导致潜在的性能下降。**QAT**在预训练或微调阶段结合了权重转换过程，从而提高了模型性能。 然而，QAT 的计算成本很高，并且需要有代表性的训练数据。  
  
　　在本文中，我们重点关注 **PTQ** 来降低参数的精度。 为了获得良好的直觉，我们将使用 GPT-2 模型进行实战演示。从最简单的量化开始，带你学会量化大模型。


> 本文代码均可在[https://github.com/17wmc/GPT-2-quantization](https://github.com/17wmc/GPT-2-quantization)此仓库中查阅。

## 1. PTQ量化原理  
### 1.1 数据类型  
　　数据类型的选择决定了所需的计算资源的数量，从而影响模型的速度和效率。 在深度学习应用中，平衡精度和计算性能成为一项至关重要的练习，因为更高的精度通常意味着更大的计算需求。

　　在各种数据类型中，浮点数主要用于深度学习，因为它们能够以高精度表示各种值。 通常，浮点数使用 n 位来存储数值。 这 n 位进一步分为三个不同的组成部分：

* **符号（Sign）**：符号位表示数字的正数或负数。 它使用一位，其中 0 表示正数，1 表示负数。
* **指数（Exponent）**：指数是一段位，表示基数（在二进制表示中通常为 2）的幂。 指数也可以是正数或负数，允许数字表示非常大或非常小的值。
* **有效数/尾数（Significand/Mantissa）**：剩余位用于存储有效数，也称为尾数。 这代表数字的有效数字。 数字的精度在很大程度上取决于有效数字的长度。

　　这种设计允许浮点数以不同的精度级别覆盖广泛的值，int 类型只包括符号位和指数位，没有尾数位。 用于这种表示的公式是：

$ （-1）^{sign}\times base^{exponent}\times significand $

如 float 9.125 在计算机中分别按照整数和尾数的二进制进行存储，9 的二进制为 1001，0.125 的二进制为 0.0010；所以 9.125 表示为 1001.0010，其二进制的科学计数法表示为 $1.001001\times 2^{3}$ 。在计算机中，任何一个数都可以表示为 $1.xxx\times 2^{n}$ 的形式，其中$n$是指数位，$xxx$ 是尾数位。可以看出**指数位决定了**该数据类型的数值**动态范围**：**指数位越多，可表示的数值范围越大**。**尾数位决定了**该数据类型的**数值精度**：**尾数位越多，可表示的数值精度越高**。深度学习中常用的数据类型如下：

![]((20241225)大语言模型量化原理和GPT-2量化实战_西西/v2-5646ace667478006162795538444321b_1440w.jpg)  
### 1.2 朴素的8位量化  
**量化本质上是数据的映射**。在图像处理中我们经常把一张 uint8 类型、数值范围在 0~255 的图片归一成 float32 类型、数值范围在 0.0~1.0 的张量，这实际就是数据的映射。在LLM中，模型的权重本质上是一个个矩阵，**模型权重的量化就是将一个个浮点数矩阵映射到其他类型的矩阵中，且这种映射是可以逆转的。**量化时通常用以下公式来进行正反变换计算：

$r=S(q-Z)\\  q=round(\frac{r}{S}+Z)\\$

其中，$S$是比例因子，表示实数和整数之间的比例关系，$Z$是零点，表示实数中的 0 经过量化后对应的整数，它们的计算方法为：

$S=\frac{r_{max}-r_{min}}{q_{max}-q_{min}}\\ Z=round(q_{max}-\frac{r_{max}}{A})\\$$r_{max}$ 、$r_{min}$分别是$r$的最大值和最小值，$q_{min}$、$q_{max}$同理。

**深度学习网络中的的计算本质上是矩阵乘法和加法**，量化操作同样使**浮点矩阵运算转换为整数矩阵运算**，假设 $r_1$、$r_2$ 是浮点实数上的两个$N \times N$的矩阵，$r_3$ 是 $r_1$、$r_2$ 相乘后的矩阵：

$r_3^{i,k}=\sum_{j=1}^N r_1^{i,j}r_2^{j,k} \\$

假设 $S_1$、$Z_1$ 是$r_1$矩阵对应的比例因子和零点， $S_2$ 、 $Z_2$ 、 $S_3$ 、 $Z_3$ 同理，那么由上式可以推出：

$S_3(q_3^{i,k}-Z_3)=\sum_{j=1}^{N}S_1(q_{1}^{i,j}-Z_1)S_2(q_2^{j,k}-Z_2)  \\$

整理一下可以得到：

$q_3^{i,k}=\frac{S_1 S_2}{S_3}\sum_{j=1}^N(q_1^{i,j}-Z_1)(q_2^{j,k}-Z_2)+Z_3 \\$

可以发现，除 $\frac{S_1 S_2}{S_3}$以外，其他都是定点整数运算。假设 $M=\frac{S_1 S_2}{S_3}$，经过大量实验统计$M$通常都是 (0, 1) 之间的实数。把 $M=\frac{S_1 S_2}{S_3}$ 代入可以得到：

$q_3^{i,k}=M\sum_{j=1}^N(q_1^{i,j}-Z_1)(q_2^{j,k}-Z_2)+Z_3=MP+Z_3 \\$

这里面$P$是量化后矩阵的运算所得。至此我们可以通过事先对M的计算，进行量化的神经网络计算。

## 2. PTQ量化实战  
### 2.1环境配置  
环境配置分为三步：

1. 确保你的电脑上至少有一张英伟达显卡，并已安装好了CUDA环境。
2. 安装Python（版本>=3.10）。
3. 安装pytorch和相关的第三方库，可以使用以下命令：


```
python -m pip install --upgrade pip
# 更换 pypi 源，加速库的安装
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers==4.74.1
pip install accelerate==1.2.1
pip install bitsandbytes==0.45.0
pip install auto-gptq==0.7.1
```
### 2.2 absmax和zeropoint量化  
**absmax量化**是一种具有**绝对最大量化**的**对称技术**，**zeropoint量化**是一种具有**零点量化**的**非对称技术**。下面用python实现这两种量化方法的int8量化：


```
import torch

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
```
通过 absmax 量化，原始数字除以张量的绝对最大值，并乘以缩放因子 (127)，以将输入映射到范围 [-127, 127]。通过零点量化，我们可以考虑不对称输入分布，例如，当考虑 ReLU 函数的输出（仅正值）时，这非常有用。 输入值首先按值的总范围 (255) 除以最大值和最小值之差进行缩放。 然后将该分布移动零点，将其映射到范围 [-128, 127]。

下面我们借助[https://cdn.openai.com/better-language-models/language\_models\_are\_unsupervised\_multitask\_learners.pdf](https://cdn.openai.com/better-language-models/language\_models\_are\_unsupervised\_multitask\_learners.pdf)来比较这两种量化方法的效果。首先加载GPT-2的模型和标记器。观察模型的大小，以便稍后进行比较并评估 8 位量化带来的内存节省。


```
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
torch.manual_seed(0)

# 设置设备为cpu
device = 'cpu'
# 加载模型和分词器
model_id = 'gpt2'
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 打印模型大小
print(f"模型的大小: {model.get_memory_footprint():,} bytes")
```
打印结果是：


```
模型的大小: 510,342,192 bytes
```
GPT2 模型的大小在 FP32 中约为 487MB。 下面使用零点和绝对最大量化来量化权重。 在下面的示例中，将应用于 GPT2 的第一个注意力层以查看结果。


```
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
```
输出结果如下：


```
原始权重:
tensor([[-0.4738, -0.2614, -0.0978,  ...,  0.0513, -0.0584,  0.0250],
        [ 0.0874,  0.1473,  0.2387,  ..., -0.0525, -0.0113, -0.0156],
        [ 0.0039,  0.0695,  0.3668,  ...,  0.1143,  0.0363, -0.0318],
        ...,
        [-0.2592, -0.0164,  0.1991,  ...,  0.0095, -0.0516,  0.0319],
        [ 0.1517,  0.2170,  0.1043,  ...,  0.0293, -0.0429, -0.0475],
        [-0.4100, -0.1924, -0.2400,  ..., -0.0046,  0.0070,  0.0198]])

Absmax量化权重:
tensor([[-21, -12,  -4,  ...,   2,  -3,   1],
        [  4,   7,  11,  ...,  -2,  -1,  -1],
        [  0,   3,  16,  ...,   5,   2,  -1],
        ...,
        [-12,  -1,   9,  ...,   0,  -2,   1],
        [  7,  10,   5,  ...,   1,  -2,  -2],
        [-18,  -9, -11,  ...,   0,   0,   1]], dtype=torch.int8)

Zero-point量化权重:
tensor([[-20, -11,  -3,  ...,   3,  -2,   2],
        [  5,   8,  12,  ...,  -1,   0,   0],
        [  1,   4,  18,  ...,   6,   3,   0],
        ...,
        [-11,   0,  10,  ...,   1,  -1,   2],
        [  8,  11,   6,  ...,   2,  -1,  -1],
        [-18,  -8, -10,  ...,   1,   1,   2]], dtype=torch.int8)
```
原始值 (FP32) 和量化值 (INT8) 之间的差异很明显，但 absmax 和零点权重之间的差异更为微妙。 在这种情况下，输入看起来偏移了 -1 值。 这表明该层的权重分布非常对称。

下面对GPT2的整个模型进行量化，并将反量化的结果保存起来。


```
import numpy as np
from copy import deepcopy

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
```
现在模型已经被量化，接下来可以检查这个过程的影响。 直观上，我们希望确保量化后的权重接近原始权重。 检查它的一种直观方法是绘制反量化权重和原始权重的分布。 如果量化是有损的，则会极大地改变权重分布。也可以比较原始模型和量化模型的性能。我们定义了一个generate\_text()函数来通过[https://mlabonne.github.io/blog/posts/2023-06-07-Decoding\_strategies.html](https://mlabonne.github.io/blog/posts/2023-06-07-Decoding\_strategies.html)生成50个token，进行模型性能比较。


```
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
```
输出结果如下：


```
原始模型回答:
The story is about: "A black woman was driving her silver Honda Civic in New York on Sunday night. She was speeding along a busy road when an officer came across an unmarked red Honda. She pulled over and spotted the two young men at the
--------------------------------------------------
Absmax量化模型回答:
The story is about:
• "The story follows a team of high school players who go in pursuit of the ultimate prize after a weekend trip to Vegas. While they arrive at Vegas they're chased down by a gang of teenagers whose sole purpose
--------------------------------------------------
Zeropoint量化模型回答:
The story is about: the first time you find yourselves in a house full of empty halls. And that's where it starts—you've got to be aware of where it's all going to get you.
```
我们可以通过计算每个输出的**困惑度（perplexity）**来比较结果，而不是试图查看一个输出是否比其他输出更有意义。 这是用于评估语言模型的常用指标，它衡量模型在预测序列中下一个标记时的不确定性。 在此比较中，我们做出共同的假设：分数越低，模型越好。 实际上，一个高度困惑的句子也可能是正确的。


```
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
```
输出结果如下：


```
原始模型困惑度:  16.95
Absmax量化模型回答困惑度:    21.40
Zeropoint量化模型回答困惑度: 25.79
```
我们看到原始模型的困惑度略低于其他两个模型。 单个实验不太可靠，但我们可以多次重复此过程以查看每个模型之间的差异。 理论上，零点量化应该比absmax稍好，但计算成本也更高。在此示例中，我们将量化技术应用于整个层（基于每个张量）。 但是，我们可以将其应用到不同的粒度级别：从整个模型到单个值。 一次性量化整个模型会严重降低性能，而量化单个值会产生很大的开销。 在实践中，我们通常更喜欢向量量化，它考虑同一张量内的行和列中值的可变性。然而，即使向量量化也不能解决**离群特征的问题**。 **异常值特征**是当模型达到一定规模（>6.7B 参数）时出现在所有 Transformer 层中的极值（负或正）。 这是一个问题，因为单个异常值可能会降低所有其他值的精度。 但放弃这些异常特征并不是一个选择，因为它会大大降低模型的性能。

### 2.3 LLM.int8() 进行 8 位量化  
　　由Dettmers 等人提出，[https://arxiv.org/abs/2208.07339](https://arxiv.org/abs/2208.07339)是异常值问题的解决方案。 它依赖于矢量方式（absmax）量化方案并引入混合精度量化。 这意味着异常值特征以 FP16 格式处理以保持其精度，而其他值以 INT8 格式处理。 由于异常值约占值的 0.1%，这有效地将 LLM 的内存占用量减少了近 2 倍。

![]((20241225)大语言模型量化原理和GPT-2量化实战_西西/v2-106359775c685f931b71b534dd4d8736_1440w.jpg)  
`LLM.int8()` 的工作原理是通过三个关键步骤进行矩阵乘法计算：

* 使用自定义阈值从输入隐藏状态 X 中提取包含异常值特征的列。
* 使用 FP16 执行异常值的矩阵乘法，使用 INT8 执行非异常值的矩阵乘法，并进行向量量化（隐藏状态 X 为行式，权重矩阵 W 为列式）。
* 对非异常值结果（INT8 到 FP16）进行反量化，并将其添加到异常值结果中，以获得 FP16 中的完整结果。

由于将bitsandbytes 库集成到Hugging Face 生态系统中，我们可以轻松使用此技术。 我们只需要在加载模型时指定`load_in_8bit=True` 。


```
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

model_int8 = AutoModelForCausalLM.from_pretrained(model_id,
                                             device_map='auto',
                                             load_in_8bit=True,
                                             )
print(f"模型大小: {model_int8.get_memory_footprint():,} bytes")
```
输出结果是：


```
模型大小: 176,527,896 bytes
```
可以发现模型现在几乎小了三倍（168MB vs. 487MB）

我们还可以使用这个量化模型生成文本并将其与原始模型进行比较。


```
text_int8 = generate_text(model_int8, "The story is about:")
print(f"原始模型回答:\n{original_text}")
print("-" * 50)
print(f"LLM.int8()量化模型回答:\n{text_int8}")
```
输出结果如下：


```
原始模型回答:
The story is about: "A black woman was driving her silver Honda Civic in New York on Sunday night. She was speeding along a busy road when an officer came across an unmarked red Honda. She pulled over and spotted the two young men at the
--------------------------------------------------
LLM.int8()量化模型回答:
The story is about: How this girl goes back to school to finish school, with a job that she hopes to someday take away from her parents. Her best friend is a classmate of hers from high school, and she's trying to make her college
```
再次，很难判断什么是最好的输出，但我们可以依靠困惑度度量来告诉我们近似的答案。


```
print(f"原始模型回答困惑度:   {ppl.item():.2f}")
ppl = calculate_perplexity(model_int8, text_int8)
print(f"LLM.int8()量化模型回答困惑度: {ppl.item():.2f}")
```
输出结果如下：


```
原始模型回答困惑度:   16.95
LLM.int8()量化模型回答困惑度: 14.49
```
在这种情况下，量化模型的困惑度是原始模型的两倍。 一般来说，情况并非如此，但它表明这种量化技术非常有竞争力。 事实上， `LLM.int8()` 的作者表明，性能下降非常低，可以忽略不计（<1%）。 然而，它在计算方面有额外的成本：对于大型模型， `LLM.int8()` 大约慢 20% 左右。

### 2.4 AutoGPTQ 量化到int4  
[https://arxiv.org/abs/2210.17323](https://arxiv.org/abs/2210.17323) 在创建可在 GPU 上高效运行的 4 位精度模型方面非常流行。下面使用 AutoGPTQ 库实现 GPTQ 算法并量化 GPT-2 模型。首先加载库并定义我们想要量化的模型--GPT2，然后要加载模型和分词器。 分词器是使用 Transformers 库中的经典 AutoTokenizer 类加载的。 另一方面，我们需要传递特定的配置（BaseQuantizeConfig）来加载模型。


```
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
```
量化过程很大程度上依赖于样本来评估和提高量化质量。 它们提供了一种比较原始模型和新量化模型产生的输出的方法。 提供的样本数量越多，进行更准确和有效比较的潜力就越大，从而提高量化质量。用 C4（Colossal Clean Crawled Corpus）数据集来生成样本。 C4 数据集是从 Common Crawl 项目收集的大规模、多语言的网络文本集合。 这个庞大的数据集经过专门清理和准备，用于训练大规模语言模型，使其成为此类任务的重要资源。 维基文本数据集是另一个流行的选择。从 C4 数据集中加载 1024 个样本，对它们进行标记并格式化。


```
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
```
现在数据集已准备就绪，我们可以开始批大小为 1 的量化过程。 完成此操作后，我们将分词器和模型保存为安全张量格式。


```
# 使用 GPTQ 进行量化
model.quantize(
    examples_ids,
    batch_size=1,
    use_triton=True,
)
# 保存模型和分词器
model.save_quantized(out_dir, use_safetensors=True)
tokenizer.save_pretrained(out_dir)
```
然后可以使用 AutoGPTQForCausalLM 和 AutoTokenizer 类从输出目录加载模型和标记生成器。


```
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# 重新加载量化模型和分词器
model = AutoGPTQForCausalLM.from_quantized(
    out_dir,
    device=device,
    use_triton=True,
    use_safetensors=True,
)
tokenizer = AutoTokenizer.from_pretrained(out_dir)
```
让我们检查一下模型是否正常工作。 AutoGPTQ 模型（大部分）作为普通transformer模型工作，这使得它与推理管道兼容，如以下示例所示：


```
# 使用模型进行文本生成
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
result = generator("The story is about:", do_sample=True, max_length=50)[0]['generated_text']
print(f"原始模型回答:\n{result}")
```
输出如下所示（可以将max\_length设置的更大，得到更多生成内容）：


```
The story is about: A child who loses his parent when trying to live a full life as a child. It's about a child in his or her mid-20s who, at age 17, falls in love with a stranger (or a
```
现在我们成功地从量化的 GPT-2 模型中获得了令人满意的完成结果。

## 参考文献  
1. Radford A, Wu J, Child R, et al. Language models are unsupervised multitask learners[J]. OpenAI blog, 2019, 1(8): 9.
2. Dettmers T, Lewis M, Belkada Y, et al. LLM. int8 (): 8-bit Matrix Multiplication for Transformers at Scale[J]. arxiv preprint arxiv:2208.07339, 2022.
3. Frantar E, Ashkboos S, Hoefler T, et al. Gptq: Accurate post-training quantization for generative pre-trained transformers[J]. arxiv preprint arxiv:2210.17323, 2022.
4. Xiao G, Lin J, Seznec M, et al. Smoothquant: Accurate and efficient post-training quantization for large language models[C]//International Conference on Machine Learning. PMLR, 2023: 38087-38099.
5. Lin J, Tang J, Tang H, et al. AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration[J]. Proceedings of Machine Learning and Systems, 2024, 6: 87-100.
6. Dettmers T, Svirschevski R, Egiazarian V, et al. Spqr: A sparse-quantized representation for near-lossless llm weight compression[J]. arXiv preprint arXiv:2306.03078, 2023.
7. Yao Z, Yazdani Aminabadi R, Zhang M, et al. Zeroquant: Efficient and affordable post-training quantization for large-scale transformers[J]. Advances in Neural Information Processing Systems, 2022, 35: 27168-27183.
