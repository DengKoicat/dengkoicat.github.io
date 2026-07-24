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

| 方案 | 压缩率 | 缓存命中率 | 综合成本 |
| :--- | :--- | :--- | :--- |
| 不压缩 | 0% | 85% | 基准（高但稳定） |
| 盲目压缩 | 30% | 15% | **更高**（缓存全废） |
| Cache-Aware 压缩 | 25% | 80% | **最低** |

Cache Breakpoint（缓存转折点）就是为了解决“既要压缩上下文，又不能破坏 Prompt Cache”这个矛盾。

它的核心思路是把上下文分成两块：

- **Breakpoint 前**：稳定前缀，尽量不动，用来提高 Prompt Cache 命中率。
- **Breakpoint 后**：动态尾部，保留最近关键消息；当动态尾部继续变长时，再把较旧部分压缩成新的稳定摘要。

需要注意，“Breakpoint 前尽量不动”不是说旧消息永远不处理，而是说不要每轮随意改写已经稳定下来的前缀。真正调整 Breakpoint 时，旧历史会先被压缩成稳定摘要，然后变成新的稳定前缀。

```text
[system prompt]                 ← 固定规则，稳定缓存
[tools schema]                  ← 工具定义，稳定缓存
[long-term memory]              ← 少量长期偏好，尽量稳定
[task summary]                  ← 旧历史压缩后的稳定摘要
--- Cache Breakpoint ---
[recent tool result 1]          ← 当前焦点区，保留
[recent tool result 2]          ← 当前焦点区，保留
[recent tool result 3]          ← 当前焦点区，保留
[current user message]          ← 最新动态，不缓存
[new assistant/tool messages]   ← 后续继续追加
```