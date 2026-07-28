---
title: "Agentic RAG"
date: 2026-07-28T13:15:00+08:00
author: "DengKoicat"
tags: ["AI", "LLM", "Agent","RL","RAG"]
categories: ["技术博客"]
readingTime: 25
toc: true
ShowToc: true
TocOpen: false
draft: false
type: "posts"
---


## Why not ReAct ? 

传统 RAG 通常只在生成前检索一次，检索内容主要由原始问题决定，因此在多跳问答中容易出现信息不足或检索偏差。ReAct、IRCoT 等方法加入了“推理—检索—再推理”的过程，使模型能够根据中间结果继续搜索。不过，这类方法仍较依赖提示词设计和模型原有的工具调用能力。当问题较复杂或检索结果存在噪声时，模型可能重复搜索、生成无效查询，或者在证据不足时提前作答。

Agentic RL 的不同之处在于，它直接对完整的推理与检索过程进行训练。模型在与搜索引擎交互的过程中，根据最终答案获得奖励，并逐渐学习何时搜索、如何拆解问题、怎样调整查询，以及什么时候停止。这比单纯规定一套操作流程更灵活，也减少了对人工编写推理轨迹的依赖。

当然，强化学习并不能完全解决检索质量和错误信息带来的问题，其效果也受到奖励设计和训练稳定性的影响。但它为 Agentic RAG 提供了一条更自然的改进路径：提示词负责规定基本交互形式，强化学习则帮助模型从实际反馈中学会更有效地搜索和推理。


## Search-R1

在 Agentic RL 范式下，Search-R1 进一步探索了强化学习驱动的搜索增强推理能力。该方法将搜索引擎建模为强化学习环境的一部分，将 LLM 自身的推理过程与外部知识检索过程进行交织，使模型能够通过 PPO、GRPO 等强化学习算法优化“思考—行动—反馈”的决策策略，而不是依赖人工设计的固定搜索流程。

为实现结构化的多轮交互，Search-R1 引入了特殊标记机制：
- <span style="color:#00AEEF">&lt;think&gt;</span> 用于表示模型内部推理过程，使模型能够逐步分析当前问题；
- <font color="#1E90FF">&lt;search&gt;</font> 用于触发搜索工具调用，使模型自主决定何时以及如何获取外部信息；
- <font color="#D4A017">&lt;information&gt;</font> 用于承载搜索引擎返回的检索内容，使外部知识能够重新融入后续推理过程；
- <font color="#D32F2F">&lt;answer&gt;</font> 用于标记最终输出结果，表示模型完成推理和检索后的答案生成阶段。

基于这种机制，模型可以形成“推理 → 检索 → 观察 → 再推理 → 回答”的循环决策模式。同时，Search-R1 采用基于最终结果的奖励函数，仅根据答案正确性优化模型行为，避免了复杂过程奖励设计带来的额外成本。因此，Search-R1 可以被视为 DeepSeek-R1 Zero 在工具增强场景下的扩展，将强化学习从单纯提升模型内部推理能力进一步拓展到检索驱动决策，使 LLM Agent 能够自主学习何时调用工具、如何利用外部信息以及如何完成复杂任务。

{{< figure
    src="ppo-grpo.png"
    caption="Fig. 1. SEARCH-R1 框架下 PPO 与 GRPO 训练过程演示。在模型推演（rollout）阶段，大语言模型（LLMs）能够与搜索引擎进行多轮交互。 (Image source: [Search-R1, 2025](https://arxiv.org/abs/2503.09516))"
    align="center"
    width="90%"
>}}



Search-R1 核心目标函数为：
$$
\max_{\pi_\theta}\ \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta(\cdot\mid x;\mathcal{R})}\left[r_\phi(x,y)\right]-\beta D_{\mathrm{KL}}\left[\pi_\theta(y\mid x;\mathcal{R})\parallel\pi_{\mathrm{ref}}(y\mid x;\mathcal{R})\right]
$$

其中，策略模型 $\pi_\theta$ 在搜索引擎 $\mathcal{R}$ 的辅助下生成交替包含“推理—检索”的轨迹，并通过奖励函数 $r_\phi$ 提高任务表现；KL 散度用于限制策略模型不要过度偏离参考模型 $\pi_{\mathrm{ref}}$。与传统仅依赖模型自身生成的 RL 方法相比，该方法显式引入外部检索信息，并使用 PPO 或 GRPO 优化检索增强推理能力。


