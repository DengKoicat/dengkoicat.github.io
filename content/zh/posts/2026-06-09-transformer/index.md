---
title: "Transformer 架构详解：从 Token 到生成"
date: 2026-06-09T12:00:00+08:00
author: "DengKoicat"
tags: ["AI", "NLP", "LLM", "Transformer", "Self-Attention", "RoPE", "Deep Learning"]
categories: ["技术博客"]
readingTime: 30
toc: true
ShowToc: true
TocOpen: false
draft: false
type: "posts"
---

Transformer 的核心贡献是把序列建模从循环结构中解放出来：每个 token 不再只能沿着时间步逐个传递信息，而是可以通过 attention 直接和上下文中的其他 token 建立联系。今天的 GPT、LLaMA、Qwen、Mistral 等 Decoder-only LLM，本质上都是在反复堆叠 Transformer Decoder Block。

本文从一次自回归语言模型的前向传播讲起：文本如何变成 token，token 如何进入 embedding，位置顺序如何通过 RoPE 注入 attention，multi-head attention 如何计算，FFN 为什么是逐 token 的非线性加工，最后模型如何得到下一个 token 的概率分布。

{{< figure
    src="transformer-paper-architecture.png"
    caption="Fig. 1. Transformer 原论文中的 Encoder-Decoder 架构图。Decoder-only LLM 可以看作保留右侧 decoder 主干，并移除 cross-attention。 (Image source: [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762))"
    align="center"
    width="70%"
>}}

## 符号

| 符号 | 含义 |
| --- | --- |
| $N$ | 输入序列长度，即 token 数 |
| $V_{\text{vocab}}$ | 词表大小 |
| $d_{\text{model}}$ | 模型隐藏维度 |
| $n_{\text{head}}$ | attention head 数量 |
| $d_k$ | 每个 head 的维度，通常 $d_k=d_{\text{model}}/n_{\text{head}}$ |
| $X$ | token embedding 矩阵，$X\in\mathbb{R}^{N\times d_{\text{model}}}$ |
| $Q,K,V$ | Query、Key、Value 矩阵 |
| $\hat{Q},\hat{K}$ | 经过 RoPE 旋转后的 Query、Key |
| $A$ | attention weight 矩阵 |
| $M$ | causal mask 矩阵 |

## Token Embedding

语言模型不能直接处理字符串。给定文本 $s$，tokenizer 会先把它变成 token id 序列：

$$
s \xrightarrow{\text{tokenizer}} [t_1,t_2,\ldots,t_N],\quad t_i\in\{1,\ldots,V_{\text{vocab}}\}.
$$

随后通过词嵌入矩阵查表：

$$
W_{\text{embed}}\in\mathbb{R}^{V_{\text{vocab}}\times d_{\text{model}}}, \quad X_i = W_{\text{embed}}[t_i].
$$

堆叠所有 token 后得到：

$$
X = \begin{bmatrix} X_1\\ X_2\\ \vdots\\ X_N \end{bmatrix} \in \mathbb{R}^{N\times d_{\text{model}}}.
$$

此时 $X_i$ 只表达“第 $i$ 个 token 是什么”，还没有表达“它在序列中的位置”。如果把 token 顺序打乱，只看 embedding 本身，attention 并不会天然知道原始顺序。

## 位置编码

原始 Transformer 使用正弦位置编码：

$$
\begin{aligned} PE_{(pos,2i)} &= \sin\left(pos / 10000^{2i/d_{\text{model}}}\right),\\ PE_{(pos,2i+1)} &= \cos\left(pos / 10000^{2i/d_{\text{model}}}\right). \end{aligned}
$$

然后把位置向量直接加到 token embedding 上：

$$
\tilde{X}=X+PE.
$$

这种做法简单直观：低维通道变化快，高维通道变化慢，不同频率共同编码位置。绝对位置编码的缺点也很明显：位置信息在输入端一次性注入，attention score 本身并不显式建模相对距离。

现代 Decoder-only LLM 更常使用旋转位置编码（Rotary Position Embedding, RoPE）。RoPE 不把位置向量加到 $X$ 上，而是在计算 attention 之前旋转 $Q$ 和 $K$。

对每个 head 内相邻两维组成的二维向量 $(x_{2i},x_{2i+1})$，位置 $m$ 对应的旋转为：

$$
\begin{bmatrix} x'_{2i}\\ x'_{2i+1} \end{bmatrix} = \begin{bmatrix} \cos(m\theta_i)&-\sin(m\theta_i)\\ \sin(m\theta_i)&\cos(m\theta_i) \end{bmatrix} \begin{bmatrix} x_{2i}\\ x_{2i+1} \end{bmatrix}, \quad \theta_i = 10000^{-2i/d_k}.
$$

记二维旋转矩阵为 $R_\theta$，则：

$$
\hat{Q}_m=R_{m\theta}Q_m,\quad \hat{K}_n=R_{n\theta}K_n.
$$

RoPE 的关键性质来自旋转矩阵的组合：

$$
R_a^T R_b = R_{b-a}.
$$

因此 Query-Key 点积可以展开为：

$$
\begin{aligned} \hat{Q}_m^T\hat{K}_n &=(R_{m\theta}Q_m)^T(R_{n\theta}K_n)\\ &=Q_m^T R_{m\theta}^T R_{n\theta}K_n\\ &=Q_m^T R_{(n-m)\theta}K_n. \end{aligned}
$$

这说明 RoPE 分别用绝对位置 $m,n$ 旋转 $Q,K$，但 attention score 中出现的是相对位移 $n-m$。这也是 RoPE 适合自回归模型的原因：它把“内容相似度”和“相对距离”一起放进了点积结构。

## Scaled Dot-Product Attention

Self-attention 的第一步是从输入表示中投影出 $Q,K,V$：

$$
Q=XW_q,\quad K=XW_k,\quad V=XW_v.
$$

其中：

$$
W_q,W_k,W_v\in\mathbb{R}^{d_{\text{model}}\times d_{\text{model}}}.
$$

如果使用 RoPE，则 attention 实际使用的是旋转后的 $\hat{Q},\hat{K}$：

$$
\hat{Q}=\text{RoPE}(Q),\quad \hat{K}=\text{RoPE}(K).
$$

对于第 $i$ 个 query 和第 $j$ 个 key，未归一化的 attention score 为：

$$
s_{ij} = \frac{\hat{Q}_i\hat{K}_j^T}{\sqrt{d_k}}.
$$

除以 $\sqrt{d_k}$ 是为了控制点积方差。若 $q,k$ 的每一维近似独立、均值为 0、方差为 1，则：

$$
\text{Var}(q^Tk) =\text{Var}\left(\sum_{\ell=1}^{d_k}q_\ell k_\ell\right) =d_k.
$$

随着 $d_k$ 增大，点积尺度会变大，softmax 更容易进入饱和区，梯度变小。缩放后：

$$
\text{Var}\left(\frac{q^Tk}{\sqrt{d_k}}\right)\approx 1.
$$

这就是 Scaled Dot-Product Attention 中 scale 的由来。

{{< figure
    src="scaled-dot-product-attention-paper.png"
    caption="Fig. 2. Scaled Dot-Product Attention：先计算 $QK^T$，再 scale、mask、softmax，最后对 $V$ 加权求和。 (Image source: [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762))"
    align="center"
    width="38%"
>}}

Decoder-only 语言模型还需要 causal mask，禁止当前位置看到未来 token。对长度为 3 的序列，score 矩阵在 mask 后变为：

$$
S_{\text{causal}} = \begin{bmatrix} s_{00} & -\infty & -\infty\\ s_{10} & s_{11} & -\infty\\ s_{20} & s_{21} & s_{22} \end{bmatrix}.
$$

再对每一行做 softmax：

$$
A_{ij} = \frac{\exp(s_{ij}+M_{ij})} {\sum_{r=0}^{N-1}\exp(s_{ir}+M_{ir})}.
$$

其中：

$$
M_{ij} = \begin{cases} 0, & j\le i,\\ -\infty, & j>i. \end{cases}
$$

最终输出为：

$$
\text{Attention}(Q,K,V) = \text{softmax} \left( \frac{\hat{Q}\hat{K}^T}{\sqrt{d_k}}+M \right)V.
$$

直觉上，$Q$ 表示“我想找什么”，$K$ 表示“我能被什么匹配”，$V$ 表示“匹配成功后我贡献什么内容”。Attention weight $A_{ij}$ 越大，说明第 $i$ 个 token 越应该从第 $j$ 个 token 处读取信息。

## Multi-Head Attention

单个 attention head 只在一个子空间中建模 token 关系。Multi-head attention 会把表示拆到多个子空间中分别计算：

$$
\text{head}_h = \text{Attention}(XW_q^{(h)},XW_k^{(h)},XW_v^{(h)}),
$$

然后拼接所有 head，并用输出矩阵混合：

$$
\text{MHA}(X) = \text{Concat}(\text{head}_1,\ldots,\text{head}_{n_{\text{head}}})W_o.
$$

{{< figure
    src="multi-head-attention-paper.png"
    caption="Fig. 3. Multi-Head Attention：多个 head 并行计算 Scaled Dot-Product Attention，再 concat 后经过线性投影。 (Image source: [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762))"
    align="center"
    width="52%"
>}}

如果 $d_{\text{model}}=12288$、$n_{\text{head}}=96$，则每个 head 的维度为：

$$
d_k = \frac{12288}{96}=128.
$$

这不是把 12288 维整体做一次 attention，而是让不同 head 在不同子空间中学习不同关系。有些 head 可能关注短距离修饰，有些可能关注长距离指代，有些可能关注格式、标点或结构边界。

{{< figure
    src="multi-head-attention-split-paper.png"
    caption="Fig. 4. 原论文中 Multi-Head Attention 的另一种画法：$Q,K,V$ 先经过线性投影并 split 到多个 head，再并行做 attention。 (Image source: [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762))"
    align="center"
    width="50%"
>}}

RoPE 通常在每个 head 内独立应用。也就是说，旋转发生在 $d_k$ 维子空间中，而不是先在完整 $d_{\text{model}}$ 上统一旋转再切分。

## Decoder Block

原始 Transformer 采用 Post-LN，即先做子层，再 residual add，再 LayerNorm。现代大模型更常使用 Pre-LN：

$$
\begin{aligned} X' &= X + \text{MHA}(\text{LN}(X)),\\ Y &= X' + \text{MLP}(\text{LN}(X')). \end{aligned}
$$

Pre-LN 的训练稳定性更好，尤其适合堆叠很多层。注意这里的 residual connection 不是细节，而是深层 Transformer 能训练起来的关键之一：它为信息和梯度提供了一条近似恒等路径。

把一个 Decoder Block 展开，可以写成：

$$
\begin{aligned} Z &= \text{LN}(X),\\ Q,K,V &= ZW_q, ZW_k, ZW_v,\\ \hat{Q},\hat{K} &= \text{RoPE}(Q),\text{RoPE}(K),\\ O &= \text{softmax}\left(\frac{\hat{Q}\hat{K}^T}{\sqrt{d_k}}+M\right)V W_o,\\ X' &= X + O,\\ Y &= X' + \text{MLP}(\text{LN}(X')). \end{aligned}
$$

其中 $M$ 是 causal mask。这个公式基本就是 Decoder-only Transformer 的核心计算。

## Feed-Forward Network

Attention 解决的是“从哪些 token 读取信息”。读完之后，每个 token 还需要对自己的隐藏表示做非线性加工，这就是 FFN / MLP 的作用。

经典 Transformer FFN 为：

$$
\text{FFN}(x)=W_2\,\sigma(W_1x+b_1)+b_2.
$$

原论文使用 ReLU：

$$
\sigma(x)=\max(0,x).
$$

现代 LLM 常用 GELU、SwiGLU 或 GEGLU。以 SwiGLU 为例：

$$
\text{SwiGLU}(x) = (xW_1)\odot \text{SiLU}(xW_3), \quad \text{SiLU}(z)=z\cdot\sigma(z).
$$

FFN 对每个位置独立应用，不直接让 token 之间通信。换句话说，token 间通信主要发生在 attention 中；FFN 负责提升每个 token 表示的非线性表达能力。

## 多层堆叠

一个 Decoder Block 只完成一轮“通信 + 加工”。实际模型会堆叠 $L$ 层：

$$
H^{(0)}=X,\quad H^{(\ell)}=\text{DecoderBlock}^{(\ell)}(H^{(\ell-1)}), \quad \ell=1,\ldots,L.
$$

最后通常再做一次 LayerNorm：

$$
H_{\text{final}}=\text{LN}(H^{(L)}).
$$

每一层都会重新计算 attention，因此浅层可能更偏局部模式，深层更偏语义和任务相关表示。最终的 $H_{\text{final},i}$ 是第 $i$ 个位置在完整历史上下文中的表示。

## 预测下一个 Token

自回归语言模型训练和推理的目标都是预测下一个 token。把最终隐藏状态投影回词表空间：

$$
\text{logits}=H_{\text{final}}W_{\text{vocab}}, \quad W_{\text{vocab}}\in\mathbb{R}^{d_{\text{model}}\times V_{\text{vocab}}}.
$$

第 $i$ 个位置得到一个 $V_{\text{vocab}}$ 维向量：

$$
\text{logits}_i\in\mathbb{R}^{V_{\text{vocab}}}.
$$

Softmax 给出下一个 token 的概率：

$$
P(t_{i+1}=v\mid t_{\le i}) = \frac{\exp(\text{logits}_{i,v})} {\sum_{u=1}^{V_{\text{vocab}}}\exp(\text{logits}_{i,u})}.
$$

训练时，模型对每个位置都预测下一个 token，并用交叉熵优化：

$$
\mathcal{L} = -\sum_{i=1}^{N-1} \log P(t_{i+1}\mid t_{\le i}).
$$

推理时，只取最后一个位置的概率分布，采样或选择一个 token，追加到上下文后继续下一轮：

$$
t_{N+1}\sim P(\cdot\mid t_{\le N}).
$$

## 总结

Decoder-only Transformer 的主线可以压缩成：

$$
\text{Token IDs} \xrightarrow{\text{Embedding}} X \xrightarrow{L\times\text{DecoderBlock}} H_{\text{final}} \xrightarrow{\text{LM Head}} \text{logits} \xrightarrow{\text{softmax}} P(\text{next token}).
$$

每个 Decoder Block 内部的核心是：

$$
\text{LN} \rightarrow Q/K/V \rightarrow \text{RoPE}(Q,K) \rightarrow \text{Causal Self-Attention} \rightarrow \text{Residual} \rightarrow \text{MLP} \rightarrow \text{Residual}.
$$

Transformer 的关键设计可以归纳为三点：

1. **Self-Attention 解决 token 间信息路由**：每个 token 通过 $QK^T$ 判断应该从哪些历史 token 读取信息。
2. **RoPE 让 attention score 感知相对位置**：位置不再简单加到 embedding 上，而是通过旋转 $Q,K$ 进入点积。
3. **FFN 对每个 token 做非线性加工**：attention 负责通信，FFN 负责变换表示，二者交替堆叠形成深层语义建模能力。

## 参考文献

[1] Vaswani, Ashish, et al. ["Attention is all you need."](https://arxiv.org/abs/1706.03762) Advances in Neural Information Processing Systems 30 (2017).

[2] Su, Jianlin, et al. ["RoFormer: Enhanced Transformer with Rotary Position Embedding."](https://arxiv.org/abs/2104.09864) Neurocomputing 568 (2024): 127063.

[3] Shazeer, Noam. ["GLU Variants Improve Transformer."](https://arxiv.org/abs/2002.05202) arXiv preprint arXiv:2002.05202 (2020).

