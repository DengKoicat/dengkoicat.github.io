---
title: "Agent Harness"
date: 2026-07-24T12:00:00+08:00
author: "DengKoicat"
tags: ["AI", "LLM", "Agent"]
categories: ["技术博客"]
readingTime: 25
toc: true
ShowToc: true
TocOpen: false
draft: false
type: "posts"
---

这篇博客主要是总结一下项目，主要包括 Harness\Context Engineering。

## Context Engineering

*为什么要有 Context Engineering*？ 核心是控制模型每一轮“看见什么、不看见什么、以什么结构看见”。

主要收益可以分成成本和质量两类：

- 节约 token，降低输出前的上下文处理时间和整体延迟
- 保持稳定前缀，尽可能命中 Prompt Cache，降低算力和调用成本
- 减少无关信息进入窗口，降低模型幻觉和注意力分散
- 把任务目标、约束、工具结果结构化，让模型知道下一步该干什么，减少瞎猜

上下文窗口的管理原则：

- 不该进入 LLM messages 的不要进：日志、trace、thread_id、request_id、调试信息等。这些属于运行时上下文，不属于模型上下文。
- 大工具结果不要全量塞进窗口：原始结果落盘，只给模型 Top-N、摘要、关键字段、`truncated` 标记和文件引用。
- 旧消息按价值处理：低价值消息直接丢弃；有决策价值但太长的历史压缩成摘要；当前用户需求、硬约束、最近关键工具结果必须保留。
- 稳定内容尽量放在前缀：system prompt、工具规则、少量长期偏好、稳定任务摘要，以提高 Prompt Cache 命中率。
- 长期记忆不要等于长聊天记录：只把可复用的用户偏好、黑名单、预算习惯等结构化存入 Store，下次按 query 召回相关记忆再注入。

### Cache Breakpoint

在上下文过长时，如果随意压缩消息，前缀就会变化，导致 Prompt Cache 失效。实测结果：

| 指标 | 不压缩 | 盲目压缩 | Cache Breakpoint |
| :--- | :--- | :--- | :--- |
| **token 数** | 80K | 56K | 60K |
| 压缩率 | 0% | 30% | 25% |
| 缓存命中率 | **85%** | 15% | **80%** |
| 实际计费 token | 12K | 47.6K | 12K |
| 综合成本 | 基准 | +297% | **-35%** |

> **关键理解**： **实际计费 token = 总 token × (1 - 缓存命中率 × 折扣)**。缓存命中的部分按半价或免费计费，所以命中率才是决定成本的关键变量。

Cache Breakpoint（缓存转折点）就是为了解决“既要压缩上下文，又不能破坏 Prompt Cache”这个矛盾。

它的核心思路是把上下文分成两块：

- **Breakpoint 前**：稳定前缀，尽量不动，用来提高 Prompt Cache 命中率。
- **Breakpoint 后**：动态尾部，保留最近关键消息；当动态尾部继续变长时，再把较旧部分压缩成新的稳定摘要。

需要注意，“Breakpoint 前尽量不动”不是说旧消息永远不处理，而是说不要每轮随意改写已经稳定下来的前缀。真正调整 Breakpoint 时，旧历史会先被压缩成稳定摘要，然后变成新的稳定前缀。

```text
[ 已压缩的早期历史摘要 ]  ← 断点之前：稳定、可缓存
--------- Breakpoint ---------
[ 最近 K 轮完整交互 ]      ← 断点之后：保持原文、不压缩
[ 当前用户消息 ]
```

### Context Window

上下文由很多部分组成：System, 长期偏好（Long-term memory）, 工作记忆, 历史对话...
Context Engineering 职责就是怎么管理这些上下文内容，最大化模型性能，节约算力资源。
最核心的原则就是---“**固定在前，动态靠后**”，保持稳定前缀，尽可能命中 Prompt Cache。
同时也要精简无用噪声/增加有用信息（to-do list，goal），让模型更加专注任务。

这个项目给的方案是，结构化上下文窗口，分为：
1. system prompt
2. 结构化任务状态
3. 相关工作记忆
4. 经 Breakpoint 处理后的 messages
5. 当前用户请求

```yaml
Context Window
├─ A. System 层 
│   ───> Agent角色与Think-Act-Observe-Reflect规则 ｜ 工具Schema与约束 ｜ 固定知识摘要
│
├─ B. 热上下文层(必进) 
│   ───> current_request(当前请求) ｜ latest_observation(最近发现) ｜ active_constraints(生效限制)
│
├─ C. 结构化任务状态层(必进) 
│   ───> goal(总目标) ｜ budget(预算) ｜ platforms(平台) ｜ current_step(当前步骤) ｜ failed_tools(失败记录)
│
├─ D. 会话工作记忆层(按需) 
│   ───> 已确认偏好 ｜ 已确认结论 ｜ 不可触碰的边界
│
├─ E. 消息历史层(Cache治理) 
│   ───> 稳定历史前缀(可缓存) ｜ 近期消息(按需截断/压缩)
│
├─ F. 当前用户输入 
└─  ───> 最后放入，永远变化
```

对于“旅行三件套，预算 300，不要塑料，亚马逊和速卖通都看看”这样的任务，上下文为：

```yaml
SYSTEM
  你是跨境购物 Agent。遵守工具调用与推荐规则。

TASK STATE
  goal: 推荐旅行三件套
  budget: <= 300 CNY
  platforms: [Amazon, AliExpress]
  current_step: price_compare
  failed_tools: []

WORKING MEMORY
  - avoid_plastic
  - prefer_niche_style
  - only_compare_direct_shipping_items

MESSAGE HISTORY
  user: 我想买旅行三件套，预算 300，不要塑料
  assistant: 已解析需求，准备搜索
  tool: Planner{...}

  ----- Cache Breakpoint -----

  assistant: 调用 item_search(Amazon)
  tool: ItemSearch{只保留名称、价格、平台、评分...}
  assistant: 调用 item_search(AliExpress)
  tool: ItemSearch{只保留名称、价格、平台、评分...}

CURRENT USER REQUEST
  继续比较亚马逊和速卖通候选商品
```

可以，把重点放在“怎么来的”：

### 拼接管理 Context

用户输入：

> 帮我找旅行三件套，预算 300 元以内，不要塑料，Amazon 和 AliExpress 都看看，优先能直邮的商品。

系统处理成三层。

1. Planner 解析用户目标

Planner 先把自然语言拆成结构化字段：

```yaml
task_state:
  goal: "推荐旅行三件套"
  budget_cny_max: 300
  platforms: ["Amazon", "AliExpress"]
  constraints:
    - "不要塑料"
    - "优先直邮"
  current_step: "search"
  failed_tools: []
```

这一步来自用户原始 Query。

2. Session Context 怎么来的

`Session Context` 是 Agent 跑任务时逐步维护出来的内部状态。

它不是一开始就完整存在，而是这样一步步来的：

```text
Planner 解析用户目标
  → 生成 task_state

Agent 调用 item_search(Amazon)
  → 工具返回 20 个候选
  → 写入 hot_context.latest_observation
  → 原始结果落盘到 cold_data.amazon_raw_result

Agent 调用 item_search(AliExpress)
  → 工具返回 18 个候选
  → 写入 hot_context.latest_observation
  → 原始结果落盘到 cold_data.aliexpress_raw_result

系统过滤塑料材质商品
  → 写入 working_memory.confirmed_decisions

系统只保留可直邮商品
  → 写入 hot_context.active_constraints

状态机发现搜索完成，要进入比价
  → 更新 task_state.current_step = price_compare
```

所以执行到比价阶段时，内部状态变成：

```yaml
task_state:
  goal: "推荐旅行三件套"
  budget_cny_max: 300
  platforms: ["Amazon", "AliExpress"]
  constraints:
    - "不要塑料"
    - "优先直邮"
  current_step: "price_compare"
  failed_tools: []

hot_context:
  latest_observation:
    - "Amazon 返回 20 个候选"
    - "AliExpress 返回 18 个候选"
  active_constraints:
    - "仅比较可直邮商品"

working_memory:
  confirmed_decisions:
    - "两平台候选都已过滤塑料材质"
    - "后续只比较价格、运费和税费"

cold_data:
  amazon_raw_result: "output/session_xxx/amazon.json"
  aliexpress_raw_result: "output/session_xxx/aliexpress.json"
```

> Session Context 是 Agent 内部的任务账本，记录当前任务已经走到哪、工具发现了什么、哪些决策已经确认、哪些原始数据被保存到文件。

3. LLM Context 怎么来的

`LLM Context` 是 `Context Manager` 在每次调用模型前，从 `Session Context` 里挑重点拼出来的。

它的拼接过程是：

```text
读取 task_state
  → 生成 TASK STATE

读取 hot_context.latest_observation
  → 生成 LATEST OBSERVATION

读取 working_memory.confirmed_decisions
  → 生成 WORKING MEMORY

读取历史消息 / checkpoint
  → 生成 STABLE HISTORY SUMMARY

读取最近工具消息
  → 生成 RECENT TOOL MESSAGES

读取 cold_data
  → 默认不展开，只保留摘要或文件引用

最后追加 current user request
  → 生成 CURRENT USER REQUEST
```

所以最终给 LLM 看的就是：

```yaml
SYSTEM
  你是跨境购物 Agent。

TASK STATE
  goal: 推荐旅行三件套
  budget: <= 300 CNY
  platforms: [Amazon, AliExpress]
  constraints: [不要塑料, 优先直邮]
  current_step: price_compare

LATEST OBSERVATION
  - Amazon 返回 20 个候选
  - AliExpress 返回 18 个候选
  - 已过滤塑料材质

WORKING MEMORY
  - 后续只比较可直邮商品
  - 重点比较价格、运费、税费和评分

STABLE HISTORY SUMMARY
  - 用户需要旅行三件套，预算 300 元以内。
  - 已完成 Amazon 和 AliExpress 的商品召回。
  - 已过滤塑料材质候选。

--------------- Cache Breakpoint ---------------

RECENT TOOL MESSAGES
  tool: Amazon Top-5 候选
  tool: AliExpress Top-5 候选

CURRENT USER REQUEST
  继续比较亚马逊和速卖通候选商品。
```

