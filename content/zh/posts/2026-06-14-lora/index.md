---
title: "PEFT: LoRA 与 QLoRA"
date: 2026-06-14
description: "Low-Rank Adaptation (LoRA) 及其量化扩展 QLoRA 的原理、推导与实现。"
slug: "lora"
author: "dengkoicat"
tags:
  - LoRA
  - QLoRA
  - PEFT
  - Fine-tuning
  - AI
categories:
  - 技术博客
readingTime: 20
toc: true
ShowToc: true
TocOpen: false
draft: false
type: "posts"
---

本文介绍两种主流的参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）方法：**LoRA** 和 **QLoRA**。LoRA 通过低秩矩阵分解大幅减少可训练参数数量；QLoRA 在此基础上引入 4-bit 量化，进一步压缩显存占用，使得在单张消费级 GPU 上微调数十亿参数的大语言模型成为可能。

## 符号

| 符号 | 含义 |
| --- | --- |
| $d$ | 隐藏层维度（Hidden Dimension） |
| $r$ | LoRA 的秩（Rank），通常 $r \ll d$ |
| $\mathbf{W} \in \mathbb{R}^{d \times d}$ | 原始预训练权重矩阵 |
| $\mathbf{A} \in \mathbb{R}^{d \times r}$ | LoRA 的下投影矩阵（Down-projection） |
| $\mathbf{B} \in \mathbb{R}^{r \times d}$ | LoRA 的上投影矩阵（Up-projection） |
| $\Delta \mathbf{W}$ | 权重更新量，$\Delta \mathbf{W} = \mathbf{B}\mathbf{A}$ |
| $\mathbf{x} \in \mathbb{R}^{d}$ | 输入向量 |
| $\alpha$ | LoRA 缩放因子（Scaling Factor） |
| $\mathcal{W}$ | 权重量化函数（Weight Quantization Function） |

## 全参数微调的问题

大语言模型（LLM）的全参数微调（Full Fine-Tuning）需要更新模型中的所有参数。以 GPT-3 为例，其 Transformer 层中的注意力权重矩阵 $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V, \mathbf{W}_O \in \mathbb{R}^{12288 \times 12288}$，每个矩阵包含约 1.5 亿参数。全参数微调存在以下问题：

- **显存占用巨大**：需要存储模型参数、梯度和优化器状态（Adam 需要额外 2 倍参数量的动量和方差），微调一个 175B 参数的模型需要超过 1TB 的 GPU 显存。
- **训练成本高昂**：多卡分布式训练的硬件和时间成本随模型规模线性增长。
- **存储开销大**：每个下游任务需要保存一份完整的模型副本。

LoRA 和 QLoRA 正是为解决上述问题而提出的。

## LoRA

### 核心思想

**LoRA（Low-Rank Adaptation）** ([Hu et al., 2022](https://arxiv.org/abs/2106.09685)) 的核心假设是：**大模型在微调过程中的权重更新量是低秩的（Low-Rank）**。因此，不需要直接更新原始的大权重矩阵 $\mathbf{W}$，而是用两个低秩矩阵 $\mathbf{A}$ 和 $\mathbf{B}$ 来近似参数更新量。

具体而言，对于预训练权重 $\mathbf{W}_0 \in \mathbb{R}^{d \times d}$，微调后的权重变为：

$$
\mathbf{W} = \mathbf{W}_0 + \Delta \mathbf{W} = \mathbf{W}_0 + \mathbf{B}\mathbf{A}
$$

其中 $\mathbf{A} \in \mathbb{R}^{d \times r}$，$\mathbf{B} \in \mathbb{R}^{r \times d}$，$r \ll d$。训练时冻结原始参数 $\mathbf{W}_0$，仅训练 $\mathbf{A}$ 和 $\mathbf{B}$。前向传播为：

$$
\mathbf{y} = \mathbf{W}_0 \mathbf{x} + \frac{\alpha}{r} \mathbf{B}\mathbf{A}\mathbf{x}
$$

其中 $\alpha$ 为缩放因子，用于控制 LoRA 分支的贡献强度；$\frac{\alpha}{r}$ 确保在改变 $r$ 时无需大幅调整其他超参数。

### 参数量对比

以 GPT-3 的 $\mathbf{W}_Q, \mathbf{W}_V \in \mathbb{R}^{12288 \times 12288}$ 为例，取 $r = 14$：

| 方法 | $\mathbf{W}_Q$ 参数量 | $\mathbf{W}_V$ 参数量 | 总参数量 | 占比 |
| --- | --- | --- | --- | --- |
| 全参数微调 | 150M | 150M | 300M | 100% |
| LoRA | 0.34M | 0.34M | 0.68M | **0.23%** |

LoRA 仅需全参数微调 0.23% 的参数量，即可达到接近的性能。

### 初始化策略

LoRA 的初始化遵循以下原则：

- **矩阵 $\mathbf{A}$**：采用高斯随机初始化（均值为 0，标准差为 0.02），保证梯度能够正常传播。
- **矩阵 $\mathbf{B}$**：初始化为零矩阵，确保训练开始时 $\Delta \mathbf{W} = \mathbf{B}\mathbf{A} = \mathbf{0}$，即模型初始行为与预训练模型完全一致。

这一初始化策略保证了 LoRA 的训练从预训练模型的输出出发，避免了随机扰动导致的训练初期不稳定。

### 训练与推理

**训练阶段**：原始权重 $\mathbf{W}_0$ 被冻结，仅更新 LoRA 参数 $\mathbf{A}$ 和 $\mathbf{B}$，前向传播为：

$$
\mathbf{y} = \mathbf{W}_0 \mathbf{x} + \mathbf{B}\mathbf{A}\mathbf{x}
$$

**推理阶段**有两种方式：

1. **保留 LoRA 分支**：推理时仍然计算 $\mathbf{W}_0 \mathbf{x} + \mathbf{B}\mathbf{A}\mathbf{x}$，适用于需要动态切换不同 LoRA 适配器的场景。
2. **权重合并**：将 LoRA 权重合并回原权重，$\mathbf{W}' = \mathbf{W}_0 + \mathbf{B}\mathbf{A}$，合并后前向传播恢复为普通线性层 $\mathbf{y} = \mathbf{W}' \mathbf{x}$，无额外推理开销。

### 代码实现

> 基于 MiniMind 的 PyTorch 实现

**Step 1：定义 LoRA 模块**

$$
\text{LoRA}(\mathbf{x}) = \mathbf{B}(\mathbf{A}\mathbf{x})
$$

```python
class LoRA(nn.Module):

    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))
```

**Step 2：插入 LoRA 到模型中**

遍历模型，找到所有 $d_{\text{in}} = d_{\text{out}}$ 的 `Linear` 层，为每个目标层添加 LoRA 分支：

$$
\mathbf{y} = \mathbf{W}_0 \mathbf{x} + \mathbf{B}\mathbf{A}\mathbf{x}
$$

```python
def apply_lora(model, rank=16):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) \
                and module.in_features == module.out_features:

            lora = LoRA(module.in_features, module.out_features, rank=rank).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            # 显式绑定，避免闭包捕获变量错误
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora
```

**Step 3：冻结原参数，仅优化 LoRA 参数**

```python
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
apply_lora(model)

# 统计参数
total_params = sum(p.numel() for p in model.parameters())
lora_params = [(n, p) for n, p in model.named_parameters() if 'lora' in n]
lora_params_count = sum(p.numel() for _, p in lora_params)

# 仅将 LoRA 参数加入优化器
optimizer = optim.AdamW([p for _, p in lora_params], lr=args.learning_rate)
```

## QLoRA

### 动机

尽管 LoRA 显著减少了可训练参数量，但在微调大模型时，模型本身的权重仍然需要以高精度（FP16/BF16）加载到显存中。例如，微调一个 65B 参数的 LLaMA 模型，即使使用 LoRA，仅加载模型权重就需要约 130GB 显存（FP16），远超单张消费级 GPU 的容量。

**QLoRA（Quantized LoRA）** ([Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)) 提出了一种在保持 LoRA 微调效果的前提下，大幅压缩模型显存占用的方法。QLoRA 的核心创新包括三个关键技术。

### 核心技术

#### 1. 4-bit NormalFloat（NF4）量化

传统的 INT4 或 FP4 量化对正态分布的权重并非最优。预训练模型的权重通常近似服从正态分布，QLoRA 为此设计了 **NF4（4-bit NormalFloat）** 数据类型：

- 对标准正态分布 $\mathcal{N}(0, 1)$ 进行分位数划分，使得每个量化区间包含相等概率质量。
- 将权重映射到最接近的 NF4 量化值，实现信息论最优的量化精度。

实验表明，NF4 量化在精度损失上显著优于传统的 FP4 和 INT4。

#### 2. 双重量化（Double Quantization）

量化过程需要为每个量化块（通常 64 个连续权重共享一个缩放因子）存储 FP32 格式的缩放因子（Scaling Factor），这会带来平均每参数约 0.5 bit 的额外开销。双重量化对这些缩放因子本身再进行一次 8-bit 量化，将额外开销降低到平均每参数约 0.127 bit。

#### 3. 分页优化器（Paged Optimizers）

利用 NVIDIA 统一内存（Unified Memory）特性，在 GPU 显存不足时自动将优化器状态卸载到 CPU 内存，类似于操作系统的虚拟内存分页机制。这避免了在长序列或大 batch 训练时因显存溢出导致的崩溃。

### QLoRA 的前向传播

QLoRA 中，预训练权重以 NF4 格式存储，但在计算时反量化为 BF16：

$$
\mathbf{y} = \text{Dequant}(\mathcal{W}^{\text{NF4}}(\mathbf{W}_0)) \cdot \mathbf{x} + \frac{\alpha}{r} \mathbf{B}\mathbf{A}\mathbf{x}
$$

其中 $\mathcal{W}^{\text{NF4}}$ 表示 NF4 量化函数，$\text{Dequant}(\cdot)$ 为反量化操作。LoRA 的 $\mathbf{A}$ 和 $\mathbf{B}$ 矩阵仍以 BF16 精度训练，梯度通过反量化操作传播到 LoRA 参数，而 **不** 更新被量化的原始权重。

### 显存对比

以 LLaMA-65B 为例：

| 方法 | 模型权重精度 | 显存占用 | 单卡可微调 |
| --- | --- | --- | --- |
| 全参数微调 | FP16 | ~780 GB | ❌ |
| LoRA | FP16 | ~130 GB | ❌ |
| QLoRA | NF4 (4-bit) | **~33 GB** | ✅（A100-40G / RTX 4090） |

QLoRA 使得在单张消费级 GPU 上微调 33B 甚至 65B 参数的模型成为可能，同时性能与 16-bit LoRA 微调相当。

### 代码实现

> 基于 Hugging Face PEFT + bitsandbytes 的典型 QLoRA 配置

**Step 1：加载 4-bit 量化模型**

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # 启用 4-bit 加载
    bnb_4bit_quant_type="nf4",              # 使用 NF4 量化类型
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时使用 BF16
    bnb_4bit_use_double_quant=True,         # 启用双重量化
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)
```

**Step 2：配置 LoRA 适配器**

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 准备模型用于量化训练（冻结量化权重，启用梯度检查点）
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,                        # LoRA 秩
    lora_alpha=32,               # 缩放因子 α
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 目标模块
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出示例: trainable params: 13,631,488 || all params: 6,751,842,304 || trainable%: 0.2018
```

**Step 3：训练**

```python
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,
        optim="paged_adamw_8bit",  # 分页优化器
    ),
    train_dataset=dataset,
)
trainer.train()
```

## 方法对比

| 方法 | 可训练参数量 | 模型权重精度 | 单卡可训练规模 | 额外推理开销 | 复杂度 |
| --- | --- | --- | --- | --- | --- |
| 全参数微调 | 100% | FP16 | ~13B | 无 | 低 |
| LoRA | ~0.1%–1% | FP16 | ~30B | 无（合并后） | 低 |
| QLoRA | ~0.1%–1% | NF4 (4-bit) | **~65B** | 无（合并后） | 中 |

LoRA 通过低秩矩阵分解将微调参数量降低至全参数的 0.2% 以下，在几乎不损失性能的前提下大幅降低了训练成本。QLoRA 在此基础上引入 NF4 量化、双重量化和分页优化器三项技术，将显存需求进一步压缩至原 LoRA 的 1/4，使得在消费级 GPU 上微调 65B 参数模型成为现实。

在实际应用中，可以根据硬件条件和任务需求选择合适的方法：
- **显存充足**：直接使用 LoRA，精度最高。
- **显存受限**：使用 QLoRA，在性能和资源之间取得平衡。
- **需要多任务部署**：保留多个 LoRA 适配器，按需切换，无需保存多个完整模型副本。

## 参考文献

[1] Hu, Edward J., et al. ["LoRA: Low-Rank Adaptation of Large Language Models."](https://arxiv.org/abs/2106.09685) *International Conference on Learning Representations (ICLR)*, 2022.

[2] Dettmers, Tim, et al. ["QLoRA: Efficient Finetuning of Quantized Language Models."](https://arxiv.org/abs/2305.14314) *Advances in Neural Information Processing Systems (NeurIPS)*, 2023.

[3] Lester, Brian, Rami Al-Rfou, and Noah Constant. ["The Power of Scale for Parameter-Efficient Prompt Tuning."](https://arxiv.org/abs/2104.08691) *Proceedings of EMNLP*, 2021.

[4] Mangrulkar, Sourab, et al. ["PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods."](https://github.com/huggingface/peft) *Hugging Face*, 2023.
