---
title: "PEFT: Parameter-Efficient Fine-Tuning"
date: 2026-06-14
description: "从提示、软提示、prefix tuning、adapter tuning 到 LoRA/QLoRA，系统梳理参数高效微调的动机、形式化定义、方法谱系与实践取舍。"
slug: "lora"
author: "dengkoicat"
tags:
  - PEFT
  - Prompting
  - Adapter
  - LoRA
  - QLoRA
  - Fine-tuning
  - LLM
categories:
  - 技术博客
readingTime: 35
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
type: "posts"
---

参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）讨论的不是“怎样把模型训练得更大”，而是一个更工程化的问题：**当基础模型已经很大、很贵、很通用时，如何用尽可能小的任务特定改动，把它稳定地适配到新的任务、领域或偏好上**。

如果只介绍 LoRA，很容易把 PEFT 理解成一种低秩矩阵技巧。但 LoRA 只是 PEFT 谱系里很成功的一支。更完整的视角应该把适配发生的位置分开：

1. **Prompting**：不改或几乎不改模型权重，只改变输入上下文；进一步发展为可训练的 soft prompt / prefix。
2. **Adapter-tuning**：冻结主干网络，在 Transformer 层内部插入小型可训练模块。
3. **LoRA**：冻结原权重，把权重更新限制在低秩子空间中；部署时还能合并回原线性层。

这三类方法的共同点是：基础模型参数 $\theta_0$ 保持共享，任务特定状态 $\phi_t$ 很小。差异在于 $\phi_t$ 被放在哪里：输入端、每层激活路径，还是权重更新路径。

{{< figure
    src="prompt-tuning-overview.png"
    caption="Prompt tuning 论文中的 Figure 1：随着 T5 模型规模增大，prompt tuning 与 full model tuning 的差距逐渐缩小。Image source: [Lester et al., 2021](https://arxiv.org/abs/2104.08691)."
    align="center"
    width="90%"
>}}

## 问题设定

预训练模型可以写成：

$$
y = f_{\theta_0}(x)
$$

其中 $\theta_0$ 是预训练得到的全部参数。全参数微调会为每个任务 $t$ 学到一套完整的新参数：

$$
\theta_t = \theta_0 + \Delta\theta_t
$$

并在训练中更新 $\theta_t$ 的全部或绝大部分参数。这个做法直接、表达能力强，但在大模型时代有几个明显问题：

- **训练显存贵**：不仅要放模型权重，还要放梯度、激活、优化器状态。Adam 类优化器通常还需要额外的一阶和二阶矩估计。
- **存储复制贵**：如果每个任务都保存一份完整模型，任务数越多，存储和分发成本越高。
- **服务切换贵**：多任务服务要么加载多个完整模型，要么频繁切换大权重。
- **遗忘风险更高**：全参数更新有更大的自由度，也更容易破坏预训练模型中已有的通用能力。

PEFT 把问题改写成：

$$
y = f(x; \theta_0, \phi_t), \quad |\phi_t| \ll |\theta_0|
$$

训练时冻结 $\theta_0$，只优化很小的任务特定参数 $\phi_t$。如果是纯 prompting，$\phi_t$ 甚至可以不是参数，而是一段人工写出的离散文本；如果是 prompt tuning、adapter 或 LoRA，$\phi_t$ 则是少量可训练张量。

这带来一个重要工程后果：**一个基础模型可以被多个小型任务模块复用**。部署时只需要保存和切换 $\phi_t$，而不是复制整个模型。

## 符号

| 符号 | 含义 |
| --- | --- |
| $\theta_0$ | 预训练基础模型参数，PEFT 中通常冻结 |
| $\phi_t$ | 任务 $t$ 的可训练适配参数 |
| $x$ | 输入 token 序列 |
| $E(x)$ | 输入 token 的 embedding 序列 |
| $d$ | Transformer 隐藏维度 |
| $L$ | Transformer 层数 |
| $m$ | prompt / prefix 的虚拟 token 数 |
| $r$ | bottleneck 或 LoRA rank，通常 $r \ll d$ |
| $W_0$ | 冻结的预训练权重矩阵 |
| $\Delta W$ | 任务适配产生的权重更新 |
| $A,B$ | LoRA 的低秩分解矩阵 |
| $\alpha$ | LoRA 分支缩放系数 |

## PEFT 的设计轴

PEFT 方法很多，但大多可以沿着四个轴理解。

| 设计轴 | 关键问题 | 典型选择 |
| --- | --- | --- |
| 适配位置 | 改输入、改激活，还是改权重更新？ | Prompting、Adapter、LoRA |
| 可训练参数量 | 每个任务要存多少参数？ | 0、$O(md)$、$O(Lrd)$、$O(rd)$ |
| 推理开销 | 适配模块是否增加额外计算路径？ | Prompt 增加序列长度；Adapter 增加 MLP；LoRA 合并后无额外延迟 |
| 可组合性 | 多个任务模块能否切换、融合或路由？ | prompt 切换最轻；adapter/LoRA 可按任务保存 |
| 表达位置 | 模型被允许在哪个空间里改变行为？ | 上下文空间、激活空间、权重子空间 |

一个有用的心智模型是：PEFT 不是简单地“少训练一些参数”，而是在给适配过程施加结构化约束。约束越强，参数越少、越容易部署，但可表达的任务变化也可能越受限制。

# Prompting

Prompting 是最轻量的适配方式。给定基础模型 $f_{\theta_0}$，我们不改变 $\theta_0$，只改变输入：

$$
y = f_{\theta_0}([\text{prompt}; x])
$$

这里的 prompt 可以是人工写出的自然语言指令、任务说明、示例、输出格式约束，或者检索系统拼接进来的上下文。严格说，普通 prompting 并不是“fine-tuning”，因为没有参数更新；但从系统功能看，它与 PEFT 的目标高度一致：让一个共享基础模型在不同任务上表现出不同条件行为。

## Hard Prompt

Hard prompt 是离散文本。典型形式包括：

- **Instruction**：描述目标、角色、约束和输出格式。
- **Few-shot examples**：在上下文中给若干输入输出样例，让模型模仿模式。
- **Chain-of-thought / rationale pattern**：通过提示诱导模型生成中间推理。
- **Structured prompt**：用 XML、Markdown、JSON schema 或固定段落组织上下文。

Hard prompt 的优点是无需训练、迭代快、可解释性强。缺点也明显：优化主要靠人工经验，prompt 对措辞敏感；上下文长度会占用推理预算；当任务需要稳定吸收大量标注数据时，离散文本很难充分利用监督信号。

从优化角度看，hard prompt 是在离散 token 空间里搜索：

$$
p^* = \arg\max_{p \in \mathcal{V}^m} J(f_{\theta_0}([p;x]))
$$

其中 $\mathcal{V}$ 是词表，$m$ 是 prompt 长度。这个搜索空间巨大且不可微，因此实践中常依赖人工、模板、检索、自动 prompt search 或 LLM 自我改写。

## Soft Prompt / Prompt Tuning

Soft prompt 把 prompt 从离散 token 变成可训练向量。设输入 embedding 为：

$$
E(x) = [e_1, e_2, \ldots, e_n], \quad e_i \in \mathbb{R}^{d}
$$

prompt tuning 学习一个长度为 $m$ 的连续 prompt：

$$
P = [p_1, p_2, \ldots, p_m], \quad p_i \in \mathbb{R}^{d}
$$

模型输入变成：

$$
[P; E(x)]
$$

训练时只更新 $P$，冻结模型其余参数。参数量约为：

$$
|\phi| = m \cdot d
$$

如果 $m=20$、$d=4096$，一个任务的 soft prompt 只有 $81{,}920$ 个参数。与数十亿参数的基础模型相比，这几乎可以忽略。

Prompt tuning 的关键发现是**规模效应**。Lester et al. (2021) 在 T5 系列上观察到：当模型规模达到数十亿参数后，prompt tuning 与 full model tuning 的差距显著缩小。这说明 soft prompt 的能力不只取决于 prompt 本身，也取决于冻结模型是否足够强，能否把输入端的连续控制信号解释成有效行为。

{{< figure
    src="prompt-params-scale.png"
    caption="Prompt tuning 论文对不同 conditioning 方法的参数量比较。Prompt tuning 只在输入端学习少量连续向量，任务特定参数极少。Image source: [Lester et al., 2021](https://arxiv.org/abs/2104.08691)."
    align="center"
    width="90%"
>}}

## Prefix Tuning

Prompt tuning 只在输入 embedding 前加虚拟 token；prefix tuning 更进一步，把可训练前缀注入到每一层 Transformer 的 attention 中。

在自注意力中，每层会产生 $K,V$：

$$
K_l = X_l W^K_l, \quad V_l = X_l W^V_l
$$

Prefix tuning 为每层学习额外的 prefix key/value：

$$
\tilde{K}_l = [P^K_l; K_l], \quad \tilde{V}_l = [P^V_l; V_l]
$$

后续 token 可以像 attending to previous tokens 一样 attending to prefix。基础模型权重仍然冻结，但每层 attention 都能看到任务特定控制向量。

{{< figure
    src="prefix-tuning-overview.png"
    caption="Prefix-tuning 论文中的示意图：prefix 被当作虚拟 token，后续 token 可以对其进行 attention。Image source: [Li & Liang, 2021](https://arxiv.org/abs/2101.00190)."
    align="center"
    width="90%"
>}}

Prefix tuning 介于 prompt tuning 和 adapter/LoRA 之间。它不像 adapter 那样改变前馈路径，也不像 LoRA 那样改变权重矩阵；但它比输入端 soft prompt 更深入，因为每一层都获得了任务特定的 key/value memory。

## Prompting 的边界

Prompting 的主要优势是轻：不需要模型副本，不需要改模型结构，容易与检索、工具调用、输出约束结合。它也非常适合任务定义经常变化的场景，例如 agent harness、RAG、数据分析和代码生成工作流。

但它的限制也很清楚：

- **上下文占用**：prompt 越长，可用于真实输入和中间推理的窗口越少。
- **稳定性问题**：离散 prompt 对措辞、示例顺序、格式噪声敏感。
- **可训练信号有限**：hard prompt 无法直接通过梯度吸收大量标注数据。
- **部署一致性**：如果 prompt 拼接逻辑依赖复杂系统状态，行为变化可能来自模型外部，而不是一个可版本化的小参数模块。

因此，在任务分布稳定、标注数据明确、需要长期复用时，adapter 和 LoRA 往往更合适。

# Adapter-Tuning

Adapter-tuning 的核心想法是：冻结 Transformer 主干，在每层插入一个小型可训练模块。原模型负责通用表示，adapter 负责把表示推向任务需要的方向。

Houlsby et al. (2019) 的 adapter 模块通常是 bottleneck 结构：

$$
\text{Adapter}(h)=W_{\text{up}}\ \sigma(W_{\text{down}}h)
$$

其中：

$$
W_{\text{down}} \in \mathbb{R}^{r \times d}, \quad
W_{\text{up}} \in \mathbb{R}^{d \times r}, \quad r \ll d
$$

adapter 输出通过残差方式加回隐藏状态：

$$
h' = h + \text{Adapter}(h)
$$

参数量约为：

$$
|\phi_{\text{adapter}}| \approx 2dr
$$

如果每层放两个 adapter，总参数量约为 $4Ldr$。当 $r$ 很小，例如 $d=4096$、$r=64$ 时，每个 adapter 约 $2 \times 4096 \times 64 \approx 0.52M$ 参数；相对整层 Transformer 的 attention 与 MLP 权重，仍然很小。

{{< figure
    src="adapter-insertion.png"
    caption="Adapter 原论文中的插入位置：在 Transformer 子层之后加入小型 adapter，并通过残差连接回主路径。Image source: [Houlsby et al., 2019](https://arxiv.org/abs/1902.00751)."
    align="center"
    width="55%"
>}}

## 为什么是 Bottleneck

Bottleneck adapter 用 $d \rightarrow r \rightarrow d$ 的结构，而不是直接学习 $d \rightarrow d$ 的全矩阵，原因是参数量和归纳偏置。

直接学习一个 $d \times d$ 变换需要 $d^2$ 参数。bottleneck 只需要：

$$
d r + r d = 2dr
$$

当 $r \ll d$ 时，节省非常明显。更重要的是，bottleneck 迫使任务更新经过低维通道，相当于假设任务适配只需要改变隐藏表示中的少数方向，而不是任意重写整个表示空间。

{{< figure
    src="adapter-architecture.png"
    caption="Adapter 模块本身的 bottleneck 结构：先降维，再非线性变换，最后升维回隐藏维度。Image source: [Houlsby et al., 2019](https://arxiv.org/abs/1902.00751)."
    align="center"
    width="55%"
>}}

## Adapter 的训练与推理

训练时：

- 冻结预训练模型参数 $\theta_0$。
- 只训练每个任务的 adapter 参数 $\phi_t$。
- 每个任务保存自己的 adapter。

推理时：

- 加载共享基础模型。
- 按任务挂载对应 adapter。
- 前向传播经过原 Transformer 子层和 adapter 子层。

这使 adapter 很适合多任务部署：基础模型只需要一份，不同任务只切换 adapter。Houlsby et al. (2019) 在 GLUE 上报告，adapter 只增加每任务约 3.6% 参数，就能达到接近 full fine-tuning 的效果。

## Adapter 的代价

Adapter 的主要代价是推理路径变长。每个 adapter 都是额外的小 MLP；即便参数很少，也会引入额外 kernel、额外激活和额外残差路径。在大 batch 或长序列推理中，这些开销可能比参数量看起来更明显。

因此 adapter 的工程取舍是：

- 如果你需要**模块化、可插拔、多任务共存**，adapter 很自然。
- 如果你需要**训练后合并、推理无额外延迟**，LoRA 往往更合适。
- 如果任务只需要轻微行为控制，soft prompt 可能比 adapter 更轻。

# LoRA

LoRA（Low-Rank Adaptation）把适配位置从激活路径移到权重更新路径。它不直接训练原权重 $W_0$，而是假设微调更新 $\Delta W$ 具有低秩结构：

$$
W = W_0 + \Delta W, \quad \Delta W = BA
$$

其中：

$$
A \in \mathbb{R}^{r \times d_{\text{in}}}, \quad
B \in \mathbb{R}^{d_{\text{out}} \times r}, \quad r \ll \min(d_{\text{in}}, d_{\text{out}})
$$

前向传播变成：

$$
y = W_0x + \frac{\alpha}{r}BAx
$$

训练时 $W_0$ 冻结，只训练 $A,B$。常见初始化是 $A$ 随机初始化、$B$ 初始化为零，因此训练开始时：

$$
\Delta W = BA = 0
$$

模型初始行为与原模型完全一致，训练过程再逐步学习任务更新。

{{< figure
    src="lora-architecture.png"
    caption="LoRA 原论文中的核心结构：冻结预训练权重 $W$，用低秩矩阵 $A,B$ 表示可训练更新。Image source: [Hu et al., 2022](https://arxiv.org/abs/2106.09685)."
    align="center"
    width="65%"
>}}

## 参数量

对一个 $d \times d$ 的线性层，全参数微调会更新：

$$
d^2
$$

个参数。LoRA 只训练：

$$
2dr
$$

个参数。两者比例为：

$$
\frac{2dr}{d^2} = \frac{2r}{d}
$$

如果 $d=4096$、$r=8$，比例约为：

$$
\frac{16}{4096} \approx 0.39\%
$$

如果只对 attention 中的 $W_Q,W_V$ 或 $W_Q,W_K,W_V,W_O$ 加 LoRA，整体可训练参数占比会更低。LoRA 论文在 GPT-3 175B 设置中强调：相对于 Adam full fine-tuning，LoRA 可把可训练参数量减少约 $10{,}000\times$，并降低训练显存需求。

## 为什么低秩有效

LoRA 的有效性来自一个经验假设：大模型预训练后已经学到丰富通用能力，下游任务不需要在所有参数方向上任意移动；它只需要在一个低维子空间中调整若干关键方向。

这个假设可以从三个角度理解：

1. **任务变化通常比预训练目标窄**。分类、抽取、格式化、风格迁移、领域 QA 等任务，往往只需要重新组合已有能力。
2. **过大的更新空间容易过拟合**。低秩约束减少自由度，可能提升低数据场景稳定性。
3. **Transformer 线性层有冗余**。attention projection 和 MLP projection 都是大矩阵，任务更新未必需要满秩。

当然，低秩不是免费午餐。rank 太小会限制表达能力；rank 太大则接近普通微调，训练和存储优势下降。实际常把 $r$、目标模块、$\alpha$ 和 dropout 一起调。

## 合并与切换

LoRA 的一个重要工程优势是可以合并。训练完成后：

$$
W' = W_0 + \frac{\alpha}{r}BA
$$

如果把 $BA$ 加回 $W_0$，推理时仍然只需要一次普通线性层计算，因此没有额外推理延迟。这一点与 adapter 不同：adapter 插入了新的计算模块，除非做结构化重写，否则推理路径会变长。

另一方面，如果需要动态切换任务，也可以不合并，而是在推理时保留 LoRA 分支。服务端可以维护一个基础模型和多个 LoRA adapter：

$$
\{\phi_1, \phi_2, \ldots, \phi_T\}
$$

按请求路由到不同 LoRA。这是 LoRA 在个性化模型、领域模型、图像生成风格包和多租户部署中很受欢迎的原因。

## LoRA 应该插在哪里

在 Transformer 中，LoRA 常见目标模块包括：

| 模块 | 作用 | 适配影响 |
| --- | --- | --- |
| $W_Q$ | 产生 query | 改变 token 关注什么 |
| $W_K$ | 产生 key | 改变 token 如何被匹配 |
| $W_V$ | 产生 value | 改变被聚合的信息内容 |
| $W_O$ | attention 输出投影 | 改变多头信息如何回到隐藏空间 |
| MLP up/down/gate projection | 前馈网络 | 改变逐 token 的非线性变换 |
| Embedding / LM head | 输入输出词表映射 | 特定词表、风格或领域可能有用 |

早期 LoRA 常只加到 $W_Q,W_V$；现代 LLM 微调经常对 attention 和 MLP 的多个 projection 都加 LoRA，尤其是在 instruction tuning 或领域适配中。

目标模块越多，表达能力越强，训练参数和显存也越高。实践中应优先从任务类型出发：

- 格式、风格、轻量分类：attention projection 往往足够。
- 领域知识、复杂指令跟随：attention + MLP 更稳。
- 词表或特殊符号变化：可能需要 embedding / LM head。

# QLoRA

LoRA 减少了可训练参数，但基础模型权重仍然要加载到显存。如果一个 65B 模型以 16-bit 权重加载，仅权重就约需要：

$$
65B \times 2\text{ bytes} \approx 130\text{ GB}
$$

QLoRA 的目标是进一步降低基础模型权重占用：把冻结的基础模型量化到 4-bit 存储，同时仍然通过 LoRA adapter 训练。

QLoRA 的前向可以概括为：

$$
y = \text{Dequant}(W_0^{\text{4-bit}})x + \frac{\alpha}{r}BAx
$$

关键是：基础模型权重以 4-bit 形式存储，计算时反量化到较高精度；梯度通过这条计算路径回传到 LoRA 参数，但不更新量化的基础权重。

{{< figure
    src="qlora-architecture.png"
    caption="QLoRA 原论文中的对比：full finetuning、LoRA 与 QLoRA 的权重、adapter、优化器状态和分页流。Image source: [Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)."
    align="center"
    width="95%"
>}}

QLoRA 的三个关键机制是：

1. **NF4（4-bit NormalFloat）**：面向近似正态分布权重设计的 4-bit 数据类型。
2. **Double Quantization**：不仅量化权重，也量化量化常数，进一步减少元数据开销。
3. **Paged Optimizers**：利用统一内存机制缓解长序列或大 batch 训练时的显存峰值。

Dettmers et al. (2023) 报告 QLoRA 可以在单张 48GB GPU 上微调 65B 参数模型，同时保持接近 16-bit 微调的任务性能。它的核心价值不在于 LoRA 公式本身变化，而在于把“冻结基础模型”这个事实用到极致：既然基础权重不更新，就可以用更激进的存储格式保存它。

# 方法对比

| 方法 | 冻结基础模型 | 可训练对象 | 参数规模 | 推理成本 | 典型优点 | 典型限制 |
| --- | --- | --- | --- | --- | --- | --- |
| Hard prompting | 是 | 无，只有文本上下文 | 0 | 增加上下文长度 | 无训练、可解释、迭代快 | 不稳定、难吸收大量监督数据 |
| Prompt tuning | 是 | 输入端 soft prompt | $O(md)$ | 增加虚拟 token | 参数极少、易切换 | 依赖模型规模，表达位置浅 |
| Prefix tuning | 是 | 每层 attention prefix | $O(Lmd)$ | attention 长度增加 | 比 prompt tuning 更深 | 序列开销更明显 |
| Adapter-tuning | 是 | 层内 bottleneck MLP | $O(Ldr)$ | 增加额外模块 | 模块化强、多任务友好 | 推理路径变长 |
| LoRA | 是 | 低秩权重更新 | $O(dr)$ 每目标矩阵 | 合并后无额外延迟 | 训练省、部署方便 | rank 和目标模块需调 |
| QLoRA | 是，且量化 | LoRA + 4-bit 基座 | 类似 LoRA | 量化反量化开销 | 显存占用极低 | 对量化实现和硬件更敏感 |
| Full fine-tuning | 否 | 全部参数 | $O(|\theta|)$ | 无额外模块 | 表达能力最强 | 训练、存储、部署成本最高 |

选择时可以用一个简单规则：

- **任务还不稳定，需求经常改**：先用 prompting。
- **有少量标注数据，希望保存极小任务状态**：试 prompt tuning 或 prefix tuning。
- **需要清晰的任务模块边界和多任务隔离**：adapter 很自然。
- **需要接近全参微调、又希望部署无额外延迟**：LoRA 是默认强基线。
- **显存是主要瓶颈**：QLoRA 通常优先于普通 LoRA。

# 与全参数微调的关系

PEFT 并不总是替代 full fine-tuning。它更像是一组不同强度的适配旋钮。

当任务与预训练分布接近，或者目标是指令风格、输出格式、领域措辞、偏好对齐时，PEFT 往往足够。因为基础模型已经具备能力，适配只是在激活这些能力。

当任务需要大规模注入新知识、改变底层推理习惯、修复模型系统性缺陷，或者数据量足够大且目标非常稳定时，全参数微调仍然可能更强。此时 PEFT 的低维约束可能成为瓶颈。

一个更实际的训练路线是：

1. 用 hard prompt 快速验证任务定义。
2. 用 LoRA/QLoRA 做低成本监督微调。
3. 如果 LoRA rank、目标模块和数据质量都已充分调优但仍明显不足，再考虑 full fine-tuning 或继续预训练。

这样可以避免一开始就把问题推到最贵的训练方式上。

# 组合与融合

PEFT 方法不是互斥的。实际系统里经常组合使用：

- **RAG + prompting**：检索提供动态知识，prompt 定义任务协议。
- **Prompt + LoRA**：prompt 控制局部行为，LoRA 固化长期偏好。
- **QLoRA + instruction tuning**：用低显存方式训练指令跟随 adapter。
- **多 LoRA 路由**：不同领域、用户或风格使用不同 LoRA。
- **Adapter fusion / LoRA merge**：多个任务模块按权重融合，得到复合能力。

组合时要注意一个问题：不同适配层可能在争夺同一行为控制权。例如 prompt 要求简洁，LoRA 训练数据偏向长解释，最终输出可能不稳定。PEFT 让模块化变容易，但模块之间仍然需要评估和版本管理。

# 常见失效模式

**1. 把 PEFT 当作数据质量的替代品。**  
PEFT 只能降低训练成本，不能修复脏数据、错误标签和混乱任务定义。LoRA 很容易学会数据里的格式偏差。

**2. rank 过小或目标模块过窄。**  
如果任务需要改 MLP 表示，却只给 $W_Q,W_V$ 加 LoRA，模型可能学得很吃力。反过来，目标模块过多也可能增加过拟合和训练成本。

**3. prompt 与训练分布不一致。**  
LoRA 训练时使用一种输入模板，推理时换成另一种模板，效果可能明显下降。PEFT 参数小，不代表它对上下文协议不敏感。

**4. 忽略推理开销。**  
adapter 参数量小，但每层额外 MLP 会增加延迟；prefix/prompt 虚拟 token 会增加 attention 长度；未合并 LoRA 也会有额外分支。

**5. 量化误差被低估。**  
QLoRA 很强，但 4-bit 基础权重、计算 dtype、量化分组、硬件 kernel 都会影响稳定性。小模型、特殊领域或数值敏感任务上需要单独验证。

**6. 评估只看单一 benchmark。**  
PEFT 可能在目标 benchmark 上很好，但在泛化、鲁棒性、拒答、安全偏好或长上下文任务上退化。尤其是 instruction tuning，需要同时评估帮助性、真实性和格式遵循。

# 小结

PEFT 的核心不是某一种技巧，而是**把任务适配从完整模型复制，改造成小型、可训练、可切换、可版本化的任务状态**。

Prompting 把适配放在上下文里，成本最低，适合快速迭代和动态任务。Adapter-tuning 把适配放在 Transformer 激活路径里，模块边界清晰，适合多任务扩展。LoRA 把适配放在权重更新的低秩子空间里，训练和部署都很实用；QLoRA 进一步利用冻结基座，把显存瓶颈压到更低。

从工程默认值看，今天的 LLM 微调常以 QLoRA/LoRA 作为起点，以 prompting 作为运行时控制层，以评估体系决定是否需要更重的训练。PEFT 最有价值的地方，正是让这些层可以分开演化：基础模型共享，任务适配轻量，部署系统按需组合。

## 引用

> Hu, Edward J., et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022. https://arxiv.org/abs/2106.09685

## 参考文献

[1] Houlsby, Neil, et al. ["Parameter-Efficient Transfer Learning for NLP."](https://arxiv.org/abs/1902.00751) ICML 2019.

[2] Li, Xiang Lisa, and Percy Liang. ["Prefix-Tuning: Optimizing Continuous Prompts for Generation."](https://arxiv.org/abs/2101.00190) ACL 2021.

[3] Lester, Brian, Rami Al-Rfou, and Noah Constant. ["The Power of Scale for Parameter-Efficient Prompt Tuning."](https://arxiv.org/abs/2104.08691) EMNLP 2021.

[4] Hu, Edward J., et al. ["LoRA: Low-Rank Adaptation of Large Language Models."](https://arxiv.org/abs/2106.09685) ICLR 2022.

[5] Dettmers, Tim, et al. ["QLoRA: Efficient Finetuning of Quantized LLMs."](https://arxiv.org/abs/2305.14314) NeurIPS 2023.

[6] Ben Zaken, Elad, Yoav Goldberg, and Shauli Ravfogel. ["BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models."](https://arxiv.org/abs/2106.10199) ACL 2022.

[7] He, Junxian, et al. ["Towards a Unified View of Parameter-Efficient Transfer Learning."](https://arxiv.org/abs/2110.04366) ICLR 2022.

[8] Mangrulkar, Sourab, et al. ["PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods."](https://github.com/huggingface/peft) Hugging Face, 2023.
