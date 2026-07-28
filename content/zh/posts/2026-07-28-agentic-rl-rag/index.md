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

Search-R1之所以出现，是因为ReAct主要依赖提示词让模型按照“思考—搜索—观察—回答”的流程调用搜索引擎，但它并没有真正优化模型的搜索策略，模型仍可能不知道什么时候该搜索、应该搜什么、搜索失败后如何改写查询以及何时停止搜索；同时，使用SFT训练ReAct还需要大量高质量的人工搜索轨迹，成本较高。Search-R1保留了类似ReAct的交互形式，但进一步将搜索引擎视为强化学习环境，通过最终答案是否正确的奖励，用PPO或GRPO直接优化整条“推理—搜索—利用结果—回答”轨迹，让模型自主学会更有效的搜索和验证策略。因此，Search-R1本质上可以理解为经过强化学习优化的ReAct式搜索框架。








