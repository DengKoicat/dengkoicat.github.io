---
title: "Agentic RL RAG"
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

ReAct 和传统的 Prompt-based 方法本质上是在“教”模型如何思考和行动，它们依赖于精心设计的提示词和固定的逻辑规则来驱动多步推理。然而，这种基于规则或提示词的决策机制存在明显的天花板：模型只是在模仿人类的推理步骤，并没有真正学会“如何更好地推理”，且容易陷入死循环或局部最优。

Agentic RL 的核心优势在于，它将模型从“规则的遵循者”变成了“策略的优化者”。通过强化学习，模型不再依赖外部设定的固定流程，而是将“思考-行动-观察”的循环内化为自身的本能。它能够在动态环境中，通过不断试错和接收环境反馈（奖励信号），自主学习并进化出最优的决策路径和工具调用策略。这种从“基于规则”到“学习优化”的范式转变，使得 Agentic RL 在适应性、搜索能力和解决复杂任务的准确率上限上，都显著超越了传统的 Prompt-based 方法。









