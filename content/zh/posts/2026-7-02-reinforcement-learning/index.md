---
title: "Reinforement Learning"
date: 2026-07-02T21:27:00+08:00
author: "dengkoicat"
tags: ["Deep Learning", "Reinforcement Learning","AI"]
categories: ["Deep Learning"]
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

## Terminology
**state** $\mathcal{s}$

**Action** $\mathcal{a}$

**Agent**: 做出动作的主题

**Policy** $\mathcal{\pi}$: 根据状态做出决策到控制 Agent 的动作（马里奥上下左右）。

**Policy function** $\pi: (s, a) \mapsto [0, 1],就是一个 *PDF* :

$$\pi(a \mid s) = \mathbb{P}(A = a \mid S = s)$$.$$


{{< figure
    src="policy.png"
    caption="Fig. 1. An example of policy. (Image source: [Shusen Wang YouTube, 2019](https://www.youtube.com/watch?v=vmkRMvhCW5c&list=PLvOO0btloRnsiqM72G4Uid0UWljikENlU/))"
    align="center"
    width="90%"
>}}





