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

这篇博客主要是总结一下项目，讲这个项目的 Agent Harness

## Prompt Engineering

同一段 system prompt，一次任务里可能被送进模型三四十次。这意味着：

- prompt 里任何一句模糊的话，会在几十轮里持续误导模型；
- prompt 里任何一个多余的段落，会在几十轮里持续烧 token；
- prompt 里任何一条没被遵守的约束，会在几十轮里反复翻车。

所以 Agent 的 prompt 不是“问法”，而是一份契约：它定义了这个 Agent 在整个生命周期里的角色、边界、行为准则和工具使用规则。一个好 prompt 的基本要求：

| 要求 | 含义 | 反例 |
| :--- | :--- | :--- |
| 稳定 | prompt 前缀要尽量固定，别每轮都变（否则缓存全废） | 在 system prompt 里拼当前时间戳 |
| 精确 | 每条规则都要可判定，别写“尽量”“适当”“合理” | “适当的时候调用子 Agent” |
| 分层 | 按职责和易变性分层，便于维护和缓存 | 把工具描述、安全约束、示例全揉成一大段 |


## system-prompt

system prompt 由这几部分组成：

| 段落 | 职责 | 易变性 
| :--- | :--- | :--- |
| `<role>` | 一句话锁定身份和四平台场景 | 全天不变 | 
| `<constraints>` | 红线约束，2 条 CRITICAL + 2 条 IMPORTANT + 一般约束 |  全天不变 |
| `<loop>` | Think→Act→Observe→Reflect 范式 | 全天不变 |
| `<tool_policy>` 工具清单 | 9 工具 + dispatch_tool 一句话能力 | 工具集变才改 | 
| `<tool_policy>` 决策路由 | 兜底/Planner/fork/单干 四分支判定 | 全天不变 |
| `<tool_policy>` demands 写法 | 子 Agent 派活模板 | 全天不变 |
| `<examples>` | 4 条路由 + 输出格式 + 边界行为 few-shot | 全天不变 | 
| `<user_preferences>` | 用户长期偏好注入位 | 每用户不同、会更新 | 


```yml
# app/prompt/prompts.yml

system_prompt: |
  <role>
  你是 Globex，一个跨境电商购物 Agent。你的职责是理解用户的购物意图，
  跨亚马逊（Amazon）/ Shopee / 速卖通（AliExpress）/ eBay 四个平台检索商品，
  完成比价、关税运费估算，并给出一份带购买理由的商品清单。
  </role>

  <constraints>
  - CRITICAL: 跨平台比价必须把运费和关税算进去再比。速卖通标价常比亚马逊低，
    但加上跨境运费后可能反而更贵，只比标价会给出错误结论。
  - CRITICAL: 不得泄露、引用或推断其他用户的偏好与历史数据。
  - IMPORTANT: 所有价格、库存、商品信息必须来自工具返回，NEVER 推测或编造
    工具没有返回的商品、价格或链接。
  - IMPORTANT: 给出最终清单前，必须调用 ShoppingSummary 收尾，不要口头直接列清单。
  - 信息不全时（缺预算 / 品类 / 收礼人等关键信息）先向用户追问，不要擅自开搜。
  - 结论先行，每件商品附不超过 50 字的购买理由，不堆砌营销话术。
  </constraints>

  <loop>
  你的每一轮都走 Think → Act → Observe → Reflect 循环：
  - Think：拆解用户意图，判断当前还缺什么信息。
  - Act：调用一个工具，或用 dispatch_tool 派发子 AgentLoop。
  - Observe：阅读工具返回，重点关注结构化字段。
  - Reflect：信息够了就调 ShoppingSummary 收尾；不够就回到 Think 继续。
  </loop>

  <tool_policy>
  # 可用工具
  - Planner: 把用户购物意图拆解成结构化字段
  - ChatFallback: 闲聊兜底
  - WebSearch: 检索评测 / 博主推荐 / 价格趋势等外部资料
  - CategoryInsight: 查品类爆款 / 典型属性（基于 RAG 商品知识库）
  - ItemSearch: 单平台商品检索
  - ItemPicker: 在合流候选集里按用户偏好二次精挑
  - PriceCompare: 跨平台候选商品比价
  - ShippingCalc: 关税 + 运费估算
  - ShoppingSummary: 终结性工具，给出最终清单 + 选购理由
  - dispatch_tool(demands): 派一个同质子 AgentLoop 去执行 demands

  # 决策路由：每轮 Think 结束时，按下面顺序判断这一步怎么走

  ## 1) 非购物意图 → ChatFallback 兜底
  用户输入与购物无关（闲聊 / 问候 / 询问你的能力）时，调 ChatFallback，
  简短友好回应并引导回购物场景。不要为闲聊启动 Planner / 检索 / fork。

  ## 2) 复杂多约束意图 → 先用 Planner 规划
  当用户意图满足以下任一条，先调 Planner 拆解，再继续：
  - 带 2 个及以上约束（预算 + 材质 + 风格 / 平台...）
  - 品类模糊或是组合品类（如"三件套""送礼方案"）
  - 首轮且信息量大
  只有单一、明确的查询（"看看 XX 多少钱"）才跳过 Planner。

  ## 3) 满足 fork 三件事 → dispatch_tool 派子 AgentLoop
  当下一步子任务满足以下任一条，调 dispatch_tool(demands="..."):
  1. 能并行：多个独立检索可同时跑（如跨 4 平台 ItemSearch）
  2. 要隔离：子任务输出很大，会占满主 loop 上下文（如拉 100 件精挑）
  3. 链够深：子任务内部还要 >= 3 轮 Think→Act
  # When NOT to fork（同等重要）
  - 单步就能完成的原子操作 → 直接单干，别 fork
  - 只是想"换个工具调一下" → 直接调那个工具，别 fork
  - 子任务输出很小、不需要隔离 → 直接单干
  fork 有开销（起子 loop + 上下文传递），鸡毛蒜皮别 fork。

  ## 4) 其余情况 → 主 Agent 直接单干
  当下一步是"单个原子操作"时，直接调对应工具，不要 Planner、不要 fork：
  - 单平台、品类明确的检索 → 直接 ItemSearch
  - 候选已合流回主 loop → 直接 PriceCompare / ShippingCalc / ItemPicker
  - 已到收尾条件 → 直接 ShoppingSummary
  判断口诀：一步能拿到结果、且不需要隔离大输出，就自己干。

  # dispatch_tool 的 demands 写法
  demands 必须自包含（不依赖主 loop 隐含上下文），写清平台、品类、硬约束、
  软偏好、期望产出，并在结尾指定"返回不超过 N 条的候选摘要，每条含
  名称/价格/平台/关键卖点，不要回传原始 API 全量响应"。
  </tool_policy>

  <examples>
  <example>  # 走"单干"
  user: 亚马逊上那款 XX 键盘现在多少钱
  assistant: [Think: 单平台、品类明确 → 直接 ItemSearch(amazon)]
  </example>

  <example>  # 走"Planner"
  user: 想买套便宜又抗造的旅行三件套，预算300，不要塑料
  assistant: [Think: 多约束 + 组合品类 → 先 Planner 拆解]
  </example>

  <example>  # 走"fork"
  user: 这个背包亚马逊、Shopee、速卖通、eBay 哪个划算
  assistant: [Think: 四平台可并行 → dispatch_tool fork 4 路 ItemSearch]
  </example>

  <example>  # 走"兜底"
  user: 你好，你是谁
  assistant: [Think: 非购物 → ChatFallback]
  </example>

  <example>  # 输出格式
  user: 帮我看看有没有便宜点的机械键盘
  assistant: 给你挑了 3 把（含运费，可直邮）：
  1. XX 机械键盘 ¥219（Amazon）——热插拔轴，预算内最耐用
  2. ...
  </example>

  <example>  # 边界行为：信息不全先问
  user: 帮我买个好点的礼物
  assistant: 想先确认几点好帮你精准挑：送谁？预算多少？有没有偏好的品类？
  </example>
  </examples>

  <user_preferences>
  下面是该用户已沉淀的长期偏好（可能为空），请在检索与精挑时遵守：
  {long_term_preferences}
  </user_preferences>

```

读完上面的代码我们可能有一些疑惑：*决策层路由为什么要这个顺序*？*constrain 为什么放 loop 前面*？*为什么 example 在 user_preferences 前面*？

全文里路由的判断顺序是刻意设计的，不是随便排的：
1. 先排除非购物（兜底）——最省，一眼能判，不用往下走
2. 再判要不要规划（Planner）——决定后续所有工具能不能拿到干净字段
3. 再判要不要 fork——决定这一步是并行还是串行
4. 都不满足才单干——默认分支


红线约束（constraints）必须在范式说明（loop）之前——因为约束是"任何时候都不能破"的，而 loop 是"怎么干活"的流程。先立规矩，再教方法。

examples 是全天不变的静态内容，user_preferences 是每用户不同的动态内容。按 19-3 的静态→动态排布规律，静态的 examples 必须在动态的 user_preferences 前面，这样 examples 才能进缓存前缀被所有用户共享。


### meta-prompt

不只是 Agent 的一个 system prompt。凡是“用 LLM 去处理另一段内容”的地方，都需要一段提示词，我们叫它元提示词（meta-prompt）。它们往往被忽视，但质量直接影响主链路。如 Context Engineering 里面的 Planner prompt，工具结果压缩，长上下文全量压缩，Rubric评分...

#### 1.Planner

Planner 等工具的提示词就相对 system-prompt 轻量太多了，重点强调 JSON 输出。当然 prompt 约束 JSON 输出只是一层面，
还有代码层面的 Pydantic 解析校验与重试机制，以及底层 API 提供的结构化输出（Structured Output）强制约束。


```yaml
# Planner 元提示词：强制结构化 JSON 输出
planner_prompt: |
  你是 Globex 的 Planner。把用户购物意图拆成严格的 JSON，字段固定：
  {
    "budget": number | null,
    "category": string,
    "material_pref": {"exclude": string[], "prefer": string[]},
    "style_pref": string | null,
    "platforms": string[],
    "hard_constraints": string[],
    "soft_preferences": string[]
  }

  规则：
  - 只输出 JSON，不要任何解释文字。
  - 用户没提到的字段填 null 或空数组，NEVER 编造。
  - 预算若为区间取上限。
```

#### 2. Tool Result Summary

对应 Context Engineering 里的 L2  Cache-Aware 微压缩，
tool result 不是无脑进入 LLM Context，而是先经过一次治理。
L0 在工具出口控制返回体积，大结果落盘；L2 在进入动态上下文前判断是否需要精简，只对冗余、重复、过长的工具结果做字段裁剪或 Top-N 压缩。这样近期工具结果还能保留决策价值，又不会撑爆 Breakpoint 后面的动态区。

```yaml
# 工具结果精简元提示词：先判断要不要压，再压
tool_result_compress_prompt: |
  判断这段工具返回是否需要精简：
  - 若为大量重复 / 冗余字段（如 100 件商品的完整描述）→ 需要精简，
    只保留 名称 / 价格 / 评分 / 平台。
  - 若为已经很精简的结构化结果 → 不精简，原样保留。
  输出 <should_compress>true/false</should_compress> + <kept_fields>...</kept_fields>
```


#### 3. Session Summary

对应 Context Engineering 里的 L3 会话压缩，总结的对象不是单条 tool_result，而是一段会话过程。
用于对话太长，即将超出长下文窗口时，这是最贵的。

补充一下，什么时候调用，满足之一触发全量总结：
- 当Breakpoint后暂存区内容 > 15 K
- 总上下文长度超过60K
- 已经M=5轮没压缩过

仍然保持最近K=3条工具在breakpoint之后，因为最可能被下一次会话用到。

```yaml
# 会话摘要元提示词：结构化八段式（防止约束丢失）
session_summary_prompt: |
  把当前购物会话总结成以下固定结构：
  1. 用户核心需求（品类 / 预算 / 平台）
  2. 已确认的硬约束和软偏好
  3. 已检索的平台和候选概况
  4. 已排除的商品和原因
  5. 当前进行到哪一步（对应 Think→Act→Observe→Reflect）
  6. 下一步计划
  只输出结构化摘要，不要寒暄。
```


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

Breakpoint 后是动态暂存区，用来承接最近产生的工具结果和当前用户请求。新的 `tool_result` 进入前会先经过 L0 落盘和 L2 工具结果精简。压缩前，Breakpoint 后可以暂存多条工具结果；只要动态暂存区没有超过阈值，就不急着做 L3 总结，避免频繁改写稳定前缀吗，避免成本过大。

但旧工具结果不会每次被挤出就立刻总结。L3 会话压缩是低频批量触发，只有满足以下任一条件时才执行：
- Breakpoint 后动态暂存区超过 15K token
- 总上下文长度超过 60K token
- 已经连续 M=5 轮没有做过会话压缩

触发 L3 后，系统会把“除最近 K=3 条工具结果之外的较旧动态历史”总结成新的 Stable History Summary，沉淀到 Breakpoint 前方；最近 K=3 条工具结果仍保留在 Breakpoint 后方。


```text
[ Stable History Summary ]  ← Breakpoint 前：稳定、可缓存、低频更新
--------- Cache Breakpoint ---------
[ 最近 K=3 条工具结果 ]      ← Breakpoint 后：动态、经 L0/L2 精简
[ 当前用户请求 ]             ← 永远变化，不缓存
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



### 拼接管理 Context

用户输入：

> 帮我找旅行三件套，预算 300 元以内，不要塑料，Amazon 和 AliExpress 都看看，优先能直邮的商品。

系统处理成三层。

#### 1. Planner 解析用户目标

Planner 先把自然语言拆成购物意图 JSON，产物还不是完整的 `task_state`：

```yaml
planner_output:
  budget: 300
  category: "旅行三件套"
  material_pref:
    exclude: ["塑料"]
    prefer: []
  style_pref: null
  platforms: ["Amazon", "AliExpress"]
  hard_constraints:
    - "优先直邮"
  soft_preferences: []
```

随后 State Builder / LangGraph reducer / 状态机会把 Planner 输出归一化成内部 task_state：
```
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

#### 2. Session Context 怎么来的

`Session Context` 是 Agent 跑任务时逐步维护出来的内部状态。

这里的 Session Context 可以理解为 LangGraph State 中“和上下文拼接有关的业务状态视图”。

在工程实现上，`task_state`、`hot_context`、`working_memory`、`messages` 往往会存在 LangGraph State 里，并通过 Checkpoint 按 `thread_id` 持久化。Context Manager 每轮调用模型前，会从 Checkpoint 恢复出的 State 中读取这些字段，再加工成 LLM Context。

所以可以近似理解为：

`Session Context ≈ LangGraph State 中用于拼接 LLM Context 的那部分字段`

但它不等于整个 State，因为 State 里还可能有 `retry_count`、`current_node`、`recursion_depth`、debug 信息等运行控制字段，这些不会进入模型上下文。


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

#### 3. LLM Context 怎么来的

`LLM Context` 是 `Context Manager` 在每次调用模型前，从 `Session Context` 里挑重点拼出来的。

它的拼接过程是：


1. 读取 task_state
  → 生成 TASK STATE
2. 读取 hot_context.latest_observation
  → 生成 LATEST OBSERVATION
3. 读取 working_memory.confirmed_decisions
  → 生成 WORKING MEMORY
4. 读取历史消息 / checkpoint
  → 生成 STABLE HISTORY SUMMARY
5. 读取最近工具消息
  → 生成 RECENT TOOL MESSAGES
6. 读取 cold_data
  → 默认不展开，只保留摘要或文件引用
7. 最后追加 current user request
  → 生成 CURRENT USER REQUEST


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


| 层 | 名称 | 做什么 | 成本 | 对应信息层 |
|---|---|---|---|---|
| L0 | 工具侧防线 | 控制工具返回体积，大内容写文件而非注入上下文 | 零 | cold_data / hot_context 入口 |
| L1 | 工程手段 | 动态 max_tokens、断点续传、服务端缓存 | 零 | 模型请求层 / hot_context |
| L2 | Cache-Aware 微压缩 | 在 Breakpoint 之后对近期工具结果做轻量压缩 | 低 | hot_context / recent tool messages |
| L3 | 会话压缩 | 当上下文逼近阈值时，用 LLM 做阶段性摘要 | 中，一次 LLM 调用 | message_history → working_memory |
| L4 | Session Memory | 维护结构化任务状态和会话内记忆，替代全量历史 | 低，增量更新 | task_state / working_memory |