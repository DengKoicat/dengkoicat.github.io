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
