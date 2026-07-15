---
title: "GPT-3 架构详解：从 Token 到生成"
date: 2026-06-09T12:00:00+08:00
author: "DengKoicat"
tags: ["AI", "NLP", "LLM", "GPT-3", "Transformer", "Self-Attention", "Deep Learning"]
categories: ["技术博客"]
readingTime: 25
toc: true
ShowToc: true
TocOpen: false
draft: false
type: "posts"
---

GPT-3 是 OpenAI 于 2020 年发布的 1750 亿参数大语言模型（[Brown et al., 2020](https://arxiv.org/abs/2005.14165)），采用 **Decoder-only Transformer** 架构。本文将从头到尾拆解 GPT-3 的一次前向传播过程：输入文本如何变成 Token，Token 如何经过 96 层 Transformer Decoder Block 的处理，最终生成下一个 Token。

## 符号

| 符号 | 含义 |
| --- | --- |
| $d_{\text{model}}$ | 模型隐藏层维度，GPT-3 中为 $12288$ |
| $d_k$ | 每个注意力头的维度，$d_k = d_{\text{model}} / n_{\text{head}} = 128$ |
| $n_{\text{head}}$ | 注意力头数量，GPT-3 中为 $96$ |
| $n_{\text{layer}}$ | Transformer Decoder Block 层数，GPT-3 中为 $96$ |
| $V_{\text{vocab}}$ | 词表大小，GPT-3 中为 $50257$ |
| $N$ | 输入序列长度（Token 数），GPT-3 中最大为 $2048$ |
| $X$ | 输入 Token Embedding 矩阵，$X \in \mathbb{R}^{N \times d_{\text{model}}}$ |
| $P$ | 位置编码矩阵，$P \in \mathbb{R}^{N \times d_{\text{model}}}$ |
| $\tilde{X}$ | 加入位置编码后的输入，$\tilde{X} = X + P$ |
| $\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v$ | Query、Key、Value 投影矩阵 |
| $\mathbf{W}_o$ | 输出投影矩阵 |
| $Q, K, V$ | Query、Key、Value 矩阵 |
| $A$ | 注意力权重矩阵（Attention Weights） |
| $\text{LN}(\cdot)$ | Layer Normalization |
| $\text{GELU}(\cdot)$ | Gaussian Error Linear Unit 激活函数 |

## Token Embedding

### 词嵌入

GPT-3 使用 [Byte-Pair Encoding (BPE)](https://arxiv.org/abs/1508.07909) 作为分词器，词表大小 $V_{\text{vocab}} = 50257$。每个 Token 通过一个可学习的嵌入矩阵映射到 $d_{\text{model}} = 12288$ 维的向量空间：

$$\mathbf{W}_{\text{embed}} \in \mathbb{R}^{V_{\text{vocab}} \times d_{\text{model}}} = \mathbb{R}^{50257 \times 12288}$$

嵌入矩阵的参数量约为：

$$|\mathbf{W}_{\text{embed}}| = 50257 \times 12288 \approx 6.17 \text{ 亿}$$

例如，输入 `"I Love You"` 经 BPE 分词后得到三个 Token ID：

| Token | ID |
| --- | --- |
| `"I"` | 101 |
| `"Love"` | 520 |
| `"You"` | 1314 |

查表后得到 Embedding 矩阵 $X \in \mathbb{R}^{3 \times 12288}$，作为下游 Transformer Decoder Block 的输入。

### 序列长度限制

由于 Self-Attention 的计算复杂度为 $O(N^2)$，GPT-3 的最大上下文窗口为 $N = 2048$ 个 Token，超出将直接报错。

> **现代模型的演进**：早期方案通过滑动窗口（Sliding Window）分批处理长文本，但重叠部分需重复计算。现代大模型（如 LLaMA 3、GPT-4）采用 **RoPE 外推** 与 **FlashAttention 算子优化**，将单次计算窗口扩大至 128k ~ 2M，实现全文本的一次性全局无损注意力计算。

## Transformer Decoder Block

GPT-3 由 $n_{\text{layer}} = 96$ 层 Transformer Decoder Block 堆叠而成。每个 Block 包含两个子模块，均采用 **Pre-LN（前置层归一化）** 架构：

$$X' = X + \text{MHA}(\text{LN}(X))$$
$$Y = X' + \text{MLP}(\text{LN}(X'))$$

其中 MHA 为多头自注意力（Multi-Head Self-Attention），MLP 为前馈网络。下面逐步拆解每个子模块。

### 位置编码（Position Embedding）

Token Embedding 只将 Token 映射到高维空间，无法感知 Token 之间的先后顺序。因此需要引入位置编码。

GPT-3 采用**可学习的绝对位置编码**：

$$P \in \mathbb{R}^{2048 \times 12288}$$

```python
self.position_embeddings = nn.Embedding(2048, 12288)
```

将 Token Embedding 与位置编码相加并归一化：

$$\tilde{X} = \text{LN}(X + P_{0:N})$$

| 位置编码方案 | 代表模型 | 特点 |
| --- | --- | --- |
| 可学习绝对位置编码 | GPT-3、BERT | 实现简单，但物理锁死窗口大小，不具备外推能力 |
| 相对位置编码 | T5、Transformer-XL | 关注相对距离，但计算复杂，难以硬件加速 |
| 旋转位置编码（RoPE） | LLaMA、Qwen、Mistral | 数学上统一绝对与相对位置，与 FlashAttention 天然适配，现代大模型标配 |

### Q、K、V 计算

Self-Attention 的核心是让每个 Token 与序列中其他 Token 交互信息。为此，将输入 $\tilde{X}$ 分别通过三个线性投影得到 Query、Key、Value：

$$Q = \tilde{X} \mathbf{W}_q, \quad K = \tilde{X} \mathbf{W}_k, \quad V = \tilde{X} \mathbf{W}_v$$

其中 $\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v \in \mathbb{R}^{12288 \times 12288}$。以 $Q$ 为例：

$$[3, 12288] \times [12288, 12288] = [3, 12288]$$

三个矩阵的直觉含义：
- **$Q$（Query）**：「我在寻找什么信息？」——当前 Token 主动试探其他 Token 的特征
- **$K$（Key）**：「我能提供什么信息？」——当前 Token 留给其他 Token 来匹配的标签
- **$V$（Value）**：「我本身的实质内容是什么？」——一旦匹配成功，实际贡献出来的语义信息

每个投影矩阵的参数量约为 $12288 \times 12288 \approx 1.51 \text{ 亿}$。

### 注意力得分与因果掩码

计算注意力得分矩阵：

$$\text{Score}_{ij} = Q_i \cdot K_j$$

对于输入 `"一个 失败的 man"`（三个 Token），得分矩阵为：

| | $K_0$（一个） | $K_1$（失败的） | $K_2$（man） |
| --- | --- | --- | --- |
| $Q_0$（一个） | $Q_0 \cdot K_0$ | $Q_0 \cdot K_1$ | $Q_0 \cdot K_2$ |
| $Q_1$（失败的） | $Q_1 \cdot K_0$ | $Q_1 \cdot K_1$ | $Q_1 \cdot K_2$ |
| $Q_2$（man） | $Q_2 \cdot K_0$ | $Q_2 \cdot K_1$ | $Q_2 \cdot K_2$ |

GPT-3 是 Decoder-only 架构，遵循**因果语言模型（Causal LM）**的约束：每个 Token 只能关注它自己和它之前的 Token，不能「偷看」未来的 Token。因此，右上角的未来位置被填上 $-\infty$：

| | $K_0$（一个） | $K_1$（失败的） | $K_2$（man） |
| --- | --- | --- | --- |
| $Q_0$（一个） | $Q_0 \cdot K_0$ | $-\infty$ | $-\infty$ |
| $Q_1$（失败的） | $Q_1 \cdot K_0$ | $Q_1 \cdot K_1$ | $-\infty$ |
| $Q_2$（man） | $Q_2 \cdot K_0$ | $Q_2 \cdot K_1$ | $Q_2 \cdot K_2$ |

直觉上，模型通过 $Q_1 \cdot K_0$ 建立了「一个 → 失败的」这种修饰关系；通过 $Q_2 \cdot K_1$ 发现「什么样的 man？→ 失败的 man」，从而获得极高的得分。

### Softmax 归一化

原始得分 $Q_i \cdot K_j \in (-\infty, +\infty)$，无法直接作为权重。需要通过 Softmax 将其归一化为概率分布：

$$A_{ij} = \text{softmax}\left(\frac{Q_i \cdot K_j}{\sqrt{d_k}}\right)$$

其中除以 $\sqrt{d_k}$ 是为了防止点积值过大导致 Softmax 进入梯度饱和区（[Vaswani et al., 2017](https://arxiv.org/abs/1706.03741)）。

Softmax 确保每行的注意力权重满足：

$$\sum_{j=0}^{i} A_{ij} = 1, \quad A_{ij} \in (0, 1)$$

### 加权求和

有了注意力权重 $A$，就可以从 $V$ 矩阵中加权提取信息。以最后一行 $Q_2$（man）为例，假设 Softmax 后的权重为 $[0.1, 0.6, 0.3]$：

$$\text{man 的新向量} = 0.1 \times V_0 + 0.6 \times V_1 + 0.3 \times V_2$$

即「man」的新表示中，60% 的信息来自「失败的」，30% 来自自身，10% 来自「一个」。完整的注意力计算公式为：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

## Multi-Head Attention

上述计算将 $12288$ 维当作一个整体。但 GPT-3 实际使用 $n_{\text{head}} = 96$ 个注意力头，每个头负责 $d_k = 12288 / 96 = 128$ 维子空间。

多头注意力的好处是：**不同 head 可以从不同角度建模 Token 之间的关系**。例如，某些 head 可能关注主谓关系，某些 head 可能关注修饰关系，某些 head 可能关注标点或格式。

在每个 head 内部，$Q, K, V \in \mathbb{R}^{3 \times 128}$，计算得到 $\text{Attention}_i \in \mathbb{R}^{3 \times 128}$。96 个 head 的结果拼接后恢复为 $[3, 12288]$。

### 输出投影（Output Projection）

96 个 head 在计算时各自独立、互不关心（Head 0 不知道 Head 1 算出了什么）。为了让不同 head 的信息发生「化学反应」，拼接后需要通过一个输出投影层：

$$O = \text{Concat}(\text{head}_1, \ldots, \text{head}_{96}) \times \mathbf{W}_o$$

```python
self.W_o = nn.Linear(12288, 12288, bias=False)
```

输出投影矩阵 $\mathbf{W}_o \in \mathbb{R}^{12288 \times 12288}$，参数量约 $1.51 \text{ 亿}$。

## 残差连接与层归一化

走完 Output Projection 后，数据不能直接送进 FFN，必须经历「复活甲」与「规范化」处理。

- **LayerNorm**：GPT-3 共 96 层，为了防止数据在矩阵乘法中数值越滚越大导致梯度爆炸，在进入 Self-Attention 和 FFN 之前都要执行 LayerNorm，将向量的均值限制为 0，方差限制为 1。

- **Residual Connection**：为了防止网络太深导致信息在纵向传递时面目全非，引入残差连接，将进入模块前的原始输入直接跨越式地硬加（Element-wise Add）到模块的输出上。

$$X' = X + \text{MHA}(\text{LN}(X))$$

## 前馈网络（FFN / MLP）

走完 Attention 之后，Token 之间已经完成了一轮信息交换。但 Attention 更像是在回答「我应该从哪些 Token 里拿信息」，而 FFN 负责对每个 Token 自己的表示进行更复杂的非线性加工。

GPT-3 的 FFN 由两层线性层和中间的 GELU 激活函数组成：

$$\text{MLP}(X') = \mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 X')$$

```python
self.fc_in  = nn.Linear(12288, 49152)   # 4x 扩展
self.fc_out = nn.Linear(49152, 12288)   # 投影回原维度
```

| 子模块 | 职责 |
| --- | --- |
| Self-Attention | Token 与 Token 之间的信息交互 |
| FFN / MLP | 每个 Token 内部特征的非线性变换 |

FFN 同样要走 Pre-LN 和残差：

$$Y = X' + \text{MLP}(\text{LN}(X'))$$

## 96 层堆叠与最终输出

GPT-3 将上述 Transformer Decoder Block 堆叠 96 层：

$$Y_1 = \text{DecoderBlock}(X)$$
$$Y_{96} = \text{DecoderBlock}(Y_{95})$$

最后一个 Block 之后，还有一层 LayerNorm：

$$H_{\text{final}} = \text{LN}(Y_{96})$$

## 预测下一个 Token

前面所有计算得到的是每个位置的最终隐藏向量 $H_{\text{final}} \in \mathbb{R}^{3 \times 12288}$。接下来需要将隐藏向量映射回词表空间：

$$\text{logits} = H_{\text{final}} \times \mathbf{W}_{\text{vocab}}$$

其中 $\mathbf{W}_{\text{vocab}} \in \mathbb{R}^{12288 \times 50257}$，因此：

$$[3, 12288] \times [12288, 50257] = [3, 50257]$$

每个位置对词表中的 $50257$ 个 Token 给出一个分数。对 logits 做 Softmax 得到概率分布：

$$P(\text{next token}) = \text{softmax}(\text{logits})$$

GPT-3 根据这个概率分布选择下一个 Token。例如给定前文：

```text
The capital of France is
```

模型在词表空间里给所有 Token 打分，其中 `" Paris"` 的概率最高。

## 总结

GPT-3 的一次前向传播流程如下：

$$\text{Token IDs} \xrightarrow{\text{Embedding}} X \xrightarrow{+P} \tilde{X} \xrightarrow{96 \times \text{DecoderBlock}} H_{\text{final}} \xrightarrow{\mathbf{W}_{\text{vocab}}} \text{logits} \xrightarrow{\text{softmax}} P(\text{next token})$$

每个 Decoder Block 内部的计算流程：

| 步骤 | 操作 | 作用 |
| --- | --- | --- |
| 1 | LayerNorm | 稳定数值分布 |
| 2 | Multi-Head Self-Attention | Token 间信息交互 |
| 3 | Output Projection | 融合多头信息 |
| 4 | Residual Connection | 保留原始信息 |
| 5 | LayerNorm | 稳定数值分布 |
| 6 | FFN / MLP | Token 内部非线性变换 |
| 7 | Residual Connection | 保留原始信息 |

GPT-3 的核心设计思想可以归结为两点：
1. **Self-Attention 解决「从哪里拿信息」**：每个 Token 通过 Q-K 匹配找到最相关的其他 Token，再通过 V 加权提取信息。
2. **FFN 解决「如何加工信息」**：对每个 Token 的表示进行独立的非线性变换，增强模型的表达能力。

## 参考文献

[1] Brown, Tom, et al. ["Language models are few-shot learners."](https://arxiv.org/abs/2005.14165) Advances in Neural Information Processing Systems 33 (2020): 1877-1901.

[2] Vaswani, Ashish, et al. ["Attention is all you need."](https://arxiv.org/abs/1706.03741) Advances in Neural Information Processing Systems 30 (2017).

[3] Sennrich, Rico, et al. ["Neural machine translation of rare words with subword units."](https://arxiv.org/abs/1508.07909) Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (2016).
