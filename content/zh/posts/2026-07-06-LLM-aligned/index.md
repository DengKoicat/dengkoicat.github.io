---
title: "大语言模型对齐"
date: 2026-07-06T09:44:00+08:00
author: "dengkoicat"
tags: ["Deep Learning", "Reinforcement Learning","LLM"]
categories: ["Deep Learning","LLM"]
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

## 前言
对齐相关概念很多，例如 **SFT**、**RLHF**、**PPO**、**DPO**、**GRPO**、**GSPO** 等。它们并不是同一层级的东西，可以先按“训练阶段”和“优化方法”来理解：

```text
预训练 Base Model
        ↓
SFT：监督微调，让模型学会按指令回答
        ↓
偏好 / 强化对齐阶段
        ├── RLHF：先训练奖励模型，再用强化学习优化模型
        │       ├── PPO：经典 RLHF 优化算法
        │       ├── GRPO：去掉 Critic 的组相对策略优化，常用于推理增强
        │       └── GSPO：序列级策略优化，进一步提升大模型 RL 训练稳定性
        │
        └── DPO：不显式训练奖励模型，也不跑在线 RL，直接用偏好数据优化模型
```

简单来说，SFT 解决“模型会不会按指令回答”，而后续的偏好对齐解决的是 “模型的回答是否更符合人类偏好或任务奖励”。

其中，RLHF 是一套对齐框架，典型流程是：先收集人类偏好数据，训练奖励模型，然后使用 PPO、GRPO、GSPO 等策略优化算法继续训练模型。PPO 是最经典的 RLHF 算法；GRPO 是 PPO 的一种变体，通过组内相对奖励估计优势，减少对价值模型 Critic 的依赖；GSPO 则把优化重点从 token 级别推进到 sequence 级别，用序列级重要性比率和裁剪来提升训练稳定性。GRPO 在 DeepSeekMath 中被提出，用于提升数学推理能力并降低 PPO 的内存开销；GSPO 则由 Qwen 团队提出，强调在大规模语言模型，尤其是 MoE 模型上的稳定性和效率。

DPO 则是另一条更简化的偏好对齐路线。它不显式训练 Reward Model，也不进行 PPO 式强化学习，而是直接利用 chosen / rejected 偏好样本，让模型提高优质回答的概率、降低劣质回答的概率。

{{<figure
    src="aligned-against-baseline.png"
    caption="Fig. 1. 可以看到 PPO,SFT 都优于 GPT(Prompt) 基线。 (Image source: [Training language models to follow instructions with human feedback ,OpenAI, 2020](https://arxiv.org/pdf/2203.02155))"
    align="center"
    width="90%"
>}}


## SFT

SFT 是 LLM 对齐训练的第一步，实际上非常简单，就是深度学习的有监督训练。经过 Pretain 的 LLM 只是一个只会预测下一个 token 的 Base Model，SFT的目的就是让这个 Base Model 变成一个可以**遵循指令**的 Instruct Model。

首先，我们需要整理一个高质量的 LLM 输出数据集，问答示例，这就是 LLM SFT 的监督源。然后 LLM 就会在 SFT 训练过程中学习按指令回答的能力。
SFT 和 Pretain 的核心都是做 next-token cross entropy，但又有区别：
1. Pretain 是全量更新 $\mathbf{W}$ ，SFT 更常用 LoRA/QLoRA
2. Pretain 是对每个 token 算 loss，SFT 只对结果（监督部分）算 loss，用户提问部分不算。因为LLM Pretain 训练后已经具备 "知识"，SFT 是教模型怎么学会对问题做回答，模型会模仿这种风格。

### SFT 数据集

训练数据本质是一个 $(x,y)$，最经典的数据就是教会模型回答自己 “身份”： 
```json
{"conversations":
	[{"role": "user", "content": "你背后的模型是哪个版本？它由谁开发？"},
	{"role": "assistant", "content": "我是由jingyaogong开发的高效小参数AI模型。"},
	{"role": "user", "content": "你模型的训练数据来源是什么？"},
	{"role": "assistant", "content": "我的训练数据涵盖多领域，确保覆盖广泛，但具体细节不公开。"}]
}
```

$$
\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{i=1}^{N} \log P_\theta(y_i|x_i)
$$

### SFT 优缺点

SFT 优势：
- 训练稳定，成本低，少量高质量样本就有成效
- 能明显提升指令遵循、对话格式和回答风格（包括，医疗等领域的回答风格）

SFT 缺点:
- 训练结果依赖数据集质量，而且全参训练可能破坏 LLM 原有能力
- 本质是模仿标准答案，不能直接学习人类偏好，对安全、伦理、复杂推理能力提升有限

## RLHF

已经有一个会回答问题的模型，如何通过奖励模型（Reward Model）的评分，让它生成更符合人类偏好的答案？这就是 RLHF 的目的。比如 LLM 知道 "1+1=2"，也知道 "一加一等于2"，但是 SFT 无法告诉模型 “1+1=2” 更好。这个时候强化学习的作用就是让模型知道 “1+1=2” 更符合人类的阅读习惯。

RLHF 是一项涉及多个模型和不同训练阶段的复杂概念，这里我们按三个步骤分解：
1. 经过 SFT 的 LLM
2. 聚合问答数据并训练一个奖励模型 (Reward Model，RM) 
3. 用强化学习 (RL) 方式微调 LM

### Bradley-Terry

Bradley-Terry 模型 是一种经典的成对比较（pairwise comparison）概率模型，用于建模在两个选项之间进行选择时的偏好概率。在 RLHF 中 Bradley-Terry 是用来训练 RM ，让偏好回答的分数更高。

假设有两个对象 $i$ 和 $j$，它们各自有一个潜在分数：$s_i, s_j$。Bradley-Terry 模型关心的不是绝对分数，而关心 $s_i - s_j$，胜负概率不由两个分数分别决定，而由**分数差**决定。

定义 $i$ 战胜 $j$ 的概率为：
$$
P(i \succ j)=\frac{\exp(s_i)}{\exp(s_i)+\exp(s_j)}
$$

这个式子也可以改写成 sigmoid 形式：
$$
P(i \succ j)=\sigma(s_i-s_j)=\frac{1}{1+\exp(-(s_i-s_j))}
$$

训练目标就是最大似然，在 RLHF 里就是让偏好回答分数更高。
给定 prompt $x$，人类更喜欢回答 $y_w$，不喜欢回答 $y_l$：

$$
y_w \succ y_l
$$
Reward Model 给回答打分：

$$
r_\phi(x,y_w),\quad r_\phi(x,y_l)
$$
Bradley-Terry 概率写成：


$$
P(y_w \succ y_l|x)=\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))
$$

对应损失：


$$
\mathcal{L}_{RM}=-\log\sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right)
$$



### PPO

LLM 建模对应 RL： 

| 符号 | 含义 |
| :--- | :--- |
|   状态 $s_t=(x,y_{\lt t})$  | prompt 加已生成 token |
|  动作 $a_t=y_t$  |  下一个 token|
| 策略 $\pi_\theta(a_t \mid s_t)$   |  当前 LLM 的 token 分布|
| 轨迹 $\tau=(x,y)$   | 一次完整生成 |
|  奖励 $r(x,y)$  | Reward Model 对完整回答打分 |


给定 prompt $x$ ，LLM 按照自回归方式生成回答：
$$
y=(y_1,\dots,y_T), \quad \pi_\theta(y|x)=\prod_{t=1}^{T}\pi_\theta(y_t|x,y_{\lt t})
$$

在 RLHF 中 LLM 被当成一个 **Policy**，所以 PPO 不是简单做的 next token 监督学习，而是在做：
$$
\max_\theta \mathbb{E}_{y\sim \pi_\theta(\cdot|x)}[r(x,y)]
$$

但直接最大化 reward 会出事：模型会钻 Reward Model 的空子，生成高分但怪异、啰嗦、谄媚或分布崩坏的回答。
因此 LLM 的 PPO 通常优化的是带 KL 约束的目标，让当前模型生成高奖励答案，同时不要偏离原来的 SFT 模型太远：
$$
\max_\theta \mathbb{E}_{y\sim \pi_\theta}
\left[
r_\phi(x,y)-\beta \, D_{\mathrm{KL}}
(\pi_\theta(\cdot \mid x)\|\pi_{\mathrm{ref}}(\cdot \mid x))
\right]
$$

- $\pi_{\mathrm{ref}}$ 通常是 SFT 后的模型
- $r_\phi(x,y)$ 是 Bradley-Terry 根据偏好数据集训练的模型，训练完成后冻结
- $\beta$ 越小，模型越追求奖励（约束小），模型变化大

现在问题来了，LLM 生成回答是逐 token 预测，当中间 token 出错往往会带来雪崩，导致后面的回答出错。现在就有了两个问题：
1. *那 KL 惩罚在 token 层怎么加呢*？
2. *完整的 KL 太贵，无法完整计算*

LLM PPO 常把 KL 写成每个 token 的惩罚。生成第 $t$ 个 token 后，近似 KL reward 可以写成：

$$
r_t^{KL}=-\beta\left[\log \pi_\theta(y_t|x,y_{\lt t})-\log \pi_{\mathrm{ref}}(y_t|x,y_{\lt t})\right]
$$

- 每生成一个 token，都检查当前策略是否比参考模型更“激进”（不要太偏离 SFT 模型）
- 回答完毕时，在加上 RM 总分（KL 不是奖励，只是一个约束项）

最终 token-level reward 常见形式是：


$$
r_t =\begin{cases} r_t^{KL}, & t \lt T \\ r_\phi(x,y)+r_t^{KL}, & t=T \end{cases}
$$

- $r_T$ 是针对这一次 prompt + response 的最终训练 reward（包含 RM 分数 + KL 惩罚）。

## 参考

[1] [Illustrating Reinforcement Learning from Human Feedback (RLHF),hugging face](https://huggingface.co/blog/zh/rlhf)

[2] [看完能和外婆解释的PPO, DPO, GRPO强化学习，zhihu](https://zhuanlan.zhihu.com/p/1984387073625593089)
