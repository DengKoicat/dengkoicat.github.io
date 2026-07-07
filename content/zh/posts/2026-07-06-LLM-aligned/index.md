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


## SFT,Supervised Fine-Tuning

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