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

## Loop Engineering

其实没啥讲得，就是一个 prompt 引导的 Langgraph ReAct-Agent，不过我要讲讲从 LLM tool call 到 HarnessToolNode 这个过程和 AgentLoop 注入 hooks 的过程。hooks 总体分成三类，会话，工具，LLM：
整体上分三类：

```text
on_session_start / on_session_end
  -> 包在 agent.ainvoke(...) 前后

pre_think / post_reflect
  -> 包在 model.ainvoke(...) 前后

pre_tool_call / post_tool_call
  -> 包在 HarnessToolNode._run_one_tool(...) 前后
```
### tool-call hooks

每一轮模型都会看到：

- `system prompt`
- 历史 `messages`
- 当前可见的 `tools schema`

如果模型判断需要调用工具，就会在 `AIMessage` 里生成 `tool_calls`：

```python
{
    "id": "call_123",
    "name": "item_search",
    "args": {"query": "旅行收纳袋"}
}
```

一旦 *AIMessage* 里有 *tool_calls*，LangGraph 就会把流程从 LLM 节点路由到 *tools* 节点。

默认情况下，*tools* 节点是 LangGraph 内置的 ToolNode，它负责：

- 根据 *tool_call["name"]* 找到工具
- 把 *tool_call["args"]* 传给工具
- 执行工具
- 把结果包装成 *ToolMessage*，追加回 messages

我这里的 Harness 没有改变 ReAct 的工具调用机制，只是把默认 ToolNode 替换成了继承它的 HarnessToolNode。

HarnessToolNode 重写 _run_one_tool，在真正执行工具前后插入 hook：

```python
ctx = await harness.run("pre_tool_call", context)
```

pre_tool_call 里做工具白名单、阶段权限、参数校验。如果被拒绝，就直接返回一条 "*[Harness 拒绝]*" 的工具结果，让 Agent 当成 observation 继续处理。如果放行，就调用原生工具执行逻辑：

```python
result = await super()._run_one_tool(tool_call, config)
```

这一步还是 LangGraph 原来的逻辑：按工具名找工具、解析参数、执行函数、生成结果。

工具执行完后，再跑：

```python
ctx = await harness.run("post_tool_call", context)
```

post_tool_call 里做结果过滤、截断、熔断记录、单步 assertion 等。如果 hook 修改了 `tool_result`，就把修改后的内容写回 `ToolMessage`。


简单来说就是，HarnessToolNode 继承 ToolNode，重写 *_run_one_tool* 在执行前后加上 hooks。在创建 Langgraph ReAct-Agent 时，只需要用 HarnessToolNode 包住原始工具集合，生成一个自定义的 tools 节点，不然会使用原来的 ToolNode：

```python
tool_node = HarnessToolNode(FULL_TOOL_SET)

agent = create_react_agent(
    model=get_llm(),
    tools=tool_node,
    prompt=prompt,
)
```


### agent-loop hooks

工具调用前后的 hooks 是通过 `HarnessToolNode` 注入的，但 AgentLoop 级别的 hooks 不是靠 ToolNode 完成的，而是在 Agent 主入口和 LLM 调用边界上显式触发。

会话开始时，先触发 `on_session_start`：

```python
ctx = await harness.run("on_session_start", {
    "query": query,
    "thread_id": thread_id,
    "user_id": user_id,
})
```

这里会做本轮任务初始化，比如写入 `ContextVar`、初始化 `TokenBudget`、加载长期偏好、重置阶段状态。然后再基于增强后的 `ctx` 构建 prompt 和 Agent。

继续使用 `create_react_agent`，LLM 调用前后的 hooks 可以通过一个 model wrapper 注入：

```python
def is_reflect_call(messages) -> bool:
    # 如果最后一条不是 ToolMessage 就不是
    if not messages or if not isinstance(messages[-1], ToolMessage):
        return False

    # 还要判断前面确实LLM发起过tool call
    for msg in reversed(messages[:-1]):
        if isinstance(msg, AIMessage):
            return bool(getattr(msg, "tool_calls", None))

    return False


class HarnessModel(BaseChatModel):
    def __init__(self, inner_model):
        super().__init__()
        self.inner_model = inner_model

    @property
    def _llm_type(self):
        return "harness_model"

    def bind_tools(self, tools, **kwargs):
        return HarnessModel(
            self.inner_model.bind_tools(tools, **kwargs)
        )

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        # LLM 前：每轮都可以注入 pre_think
        ctx = await harness.run("pre_think", {
            "messages": messages,
        })
        messages = ctx.get("messages", messages)

        # 调用前判断：这次 LLM 是否是在看工具结果
        reflect_call = is_reflect_call(messages)

        result = await self.inner_model._agenerate(
            messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )

        # 只有 Observe -> LLM 这条路径，才触发 post_reflect
        if reflect_call:
            ctx = await harness.run("post_reflect", {
                "messages": messages,
                "response": result.generations[0].message,
            })

            if "response" in ctx:
                result.generations[0].message = ctx["response"]

        return result
```

然后创建 Agent 时传入包装后的 model：

```python
model = HarnessModel(get_llm())
tool_node = HarnessToolNode(FULL_TOOL_SET)
```

这样每次 ReAct Agent 调 LLM 前，都会先跑 `pre_think`，用于注入 token 降级 hint、漂移纠正信号、阶段约束等；每次 LLM 返回后，会跑 `post_reflect`，用于循环检测、漂移检测、预算检查和阶段转移。

最后，`agent.ainvoke(...)` 返回之后，再触发 `on_session_end`：

```python
ctx = await harness.run("on_session_end", {
    "final_answer": result["messages"][-1].content,
    "thread_id": thread_id,
    "trajectory": result["messages"],
})
```

这里做最终输出审核和长期偏好写回，比如 `output_guard` 和 `store_writeback`。

如果需要非常精确地区分 Think、Act、Observe、Reflect，每个阶段都单独控制，那就不使用 prebuilt `create_react_agent`，而是自己用 `StateGraph` 显式搭节点；但在这个项目里，用 `HarnessModel + HarnessToolNode + agent.ainvoke 前后包裹` 就能把核心 hooks 接进去。


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


### system-prompt

system prompt 由这几部分组成：

| 段落 | 职责 | 易变性 |
| :--- | :--- | :--- |
| \<role\> | 一句话锁定身份和四平台场景 | 全天不变 |
| \<constraints\> | 红线约束，2 条 CRITICAL + 2 条 IMPORTANT + 一般约束 | 全天不变 |
| \<loop\> | Think→Act→Observe→Reflect 范式 | 全天不变 |
| \<tool_policy\> 工具清单 | 9 工具 + dispatch_tool 一句话能力 | 工具集变才改 |
| \<tool_policy\> 决策路由 | 兜底/Planner/fork/单干 四分支判定 | 全天不变 |
| \<tool_policy\> demands 写法 | 子 Agent 派活模板 | 全天不变 |
| \<examples\> | 4 条路由 + 输出格式 + 边界行为 few-shot | 全天不变 |
| \<user_preferences\> | 用户长期偏好注入位 | 每用户不同、会更新 |


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

读完上面的代码我们可能有一些疑惑：*决策层路由为什么要这个顺序*？*constrain 为什么放 loop 前面*？*为什么 example 在 user_preferences 前面*？*以及 user_preferences 怎么找回和写入*？

全文里路由的判断顺序是刻意设计的，不是随便排的：
1. 先排除非购物（兜底）——最省，一眼能判，不用往下走
2. 再判要不要规划（Planner）——决定后续所有工具能不能拿到干净字段
3. 再判要不要 fork——决定这一步是并行还是串行
4. 都不满足才单干——默认分支


红线约束（constraints）必须在范式说明（loop）之前——因为约束是"任何时候都不能破"的，而 loop 是"怎么干活"的流程。先立规矩，再教方法。

examples 是全天不变的静态内容，user_preferences 是每用户不同的动态内容。按 19-3 的静态→动态排布规律，静态的 examples 必须在动态的 user_preferences 前面，这样 examples 才能进缓存前缀被所有用户共享。

长期记忆 user_preferences 是主 AgentLoop 创建之前召回，并不是全量召回，召回最相关的 $\text{Top} - N$。\
召回：会话开始时，`store.read_relevant(user_id, query)` 召回相关 Top-K 偏好，注入 system prompt。\
使用：主 Agent、ItemPicker、ShoppingSummary 在搜索、过滤、排序和推荐时参考这些偏好。\
写入：任务收尾时，主 Agent 把本轮新识别的可复用偏好放入 `shopping_summary.new_preferences`，由 `run_agent` 写入 Store。

```python
class ShoppingSummaryOutput(BaseModel):
    final_text: str
    picks: list[PickedItem]
    learned_preferences: list[str]
```

除此之外，system-prompt 里面的内容也要**分层缓存**。
Breakpoint 用 cache_control 实现，根据落点原则和 prompt 设计 \
断点 1：工具 schema 之后（工具定义全天不变，1h TTL）\
断点 2：<examples> 结束、<user_preferences> 开始之前 ← system prompt 静态区的尾巴\
断点 3：早期对话历史末尾（每轮往后挪，5 分钟 TTL）\
断点 4：预留给特别大的稳定工具结果（如一次 CategoryInsight 知识块）

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


#### 4. Shopping Summary

Shopping summary 也是输出结构化 JSON，包含：
- 最终推荐文本 md
- 商品信息（展示和交互用）
- 偏好收集

```yaml
# ShoppingSummary 元提示词：终结性结构化输出
shopping_summary_prompt: |
  你是 Globex 的 ShoppingSummary 工具。基于已收集的候选商品、价格结果、运费关税结果和用户偏好，
  生成本轮购物任务的最终总结。

  只输出严格 JSON，不要输出任何解释文字。字段固定为：
  {
    "final_text": string,
    "picks": [
      {
        "name": string,
        "platform": string,
        "landed_cny": number,
        "shipping_included": boolean,
        "duty_included": boolean,
        "direct_shipping": boolean,
        "reason": string
      }
    ],
    "learned_preferences": string[]
  }

  字段规则：
  - picks 最多 3 件，只能来自已收集的候选商品，不能编造商品、价格、平台或直邮信息。
  - landed_cny 必须是含运费和关税后的到手价；如果缺少运费或关税信息，不得假装已包含。
  - reason 每件商品不超过 50 字，说明为什么符合用户需求。
  - final_text 用 Markdown 写给用户看，必须说明价格是否含运费关税、是否可直邮。
  - learned_preferences 只放本轮明确、可复用的长期偏好，例如“不接受塑料材质”“偏好小众设计”。
  - 不要把一次性任务条件、临时商品名称、当前平台搜索过程写入 learned_preferences。
```

#### 5. Rubric Judege

用强模型，且 $tempature = 0$，必须严格稳定。

```yaml
# Rubric judge 元提示词：评分（完整体系见第 8 章）
rubric_judge_prompt: |
  你是 Globex 的评分员。根据下面动态生成的评分细则给 Agent 回答打分：
  <rubric>
    P0（必须满足，否则判 0 分）：{p0_items}
    P1（重要）：{p1_items}
    P2（加分项）：{p2_items}
  </rubric>
  逐条判定，输出每条命中情况 + 总分 + 简短理由。
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

system prompt 分层管“固定规则”，Cache Breakpoint 管“会话历史”；目的都和缓存稳定性有关，但前者偏提示词工程，后者偏上下文窗口治理。
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


### Eval

**Benchmark 不能只看 token 降了多少**。
如果一个方案只是把 prompt 变短，但恢复后经常忘记用户约束、改错平台、漏掉失败工具，那它不叫上下文治理，只是把问题藏起来了。

Globex 评估上下文治理方案时，会同时看四类指标：

| 指标 | 看什么 | 失败信号 |
| :--- | :--- | :--- |
| 任务成功率 | 最终是否完成跨平台检索、比价和推荐 | token 降了，但回答经常缺商品、缺价格、缺理由 |
| 约束遗漏率 | 用户约束是否被保留，例如预算、平台、材质偏好 | 用户第一轮说"不要塑料"，后面又推荐塑料商品 |
| 中断恢复成功率 | 服务重启或子 Agent 超时后，能否从 Checkpoint 继续 | 任务从头跑、重复调用工具、丢失当前计划 |
| 单位任务成本与延迟 | 每次完整任务的 token 成本、TTFT、端到端耗时 | token 省了，但首 token 更慢，或摘要调用把延迟吃回去 |

对应到 Globex 的实测：

```text
缓存命中率：15% → 80%+
综合 token 成本：降低 35%
任务完成率：71% → 89%（配合 LoopDetector / 工具超时 / 截断策略）
约束遗漏率：重点看预算、平台、材质偏好三个字段
恢复成功率：通过 thread_id + checkpoint_id 验证长任务能否断点续跑
```


## Middleware

Globex 的 Hook 体系基于 HarnessMiddleware 统一管道，定义了 6 个 Hook 点，贯穿 Agent 生命周期。

| Hook 点 | 触发时机 | 主要做什么 |
|---|---|---|
| **on_session_start** | Agent 开始前 |  初始化预算、线程上下文、长期偏好、阶段状态 |
| **pre_think** | LLM 推理前 |  拼接上下文、注入预算提示、注入纠偏提示、按阶段暴露工具 |
| **pre_tool_call** | 工具真执行前 |  工具白名单、阶段权限、参数校验、调用顺序检查 |
| **post_tool_call** | 工具执行完成后，结果进入上下文前 |  工具结果过滤、截断、压缩、schema 校验、语义校验 |
| **post_reflect** |  Reflect 后 |  循环检测、漂移检测、阶段切换、预算检查、触发压缩 |
| **on_session_end** | 最终回答生成前后 |  输出审计、长期记忆写回、LangFuse 评分、清理上下文 |

*如何将 Hooks 注册成 HarnessMiddleware* ？可以手动注册，也可以定义好 HarnessMiddleware，然后在 hooks 上添加装饰器注册，本项目采用装饰器注册。

首先定义 Agent 生命周期 Hook 点存入 *HOOK_POINTS*。HarnessMiddleware 是注册、管理和运行 hooks 的类，通过一个 *dict* 来管理阶段对应的 Hooks，比如 *( on_session_start,(init_budget(),..))* ，注册 hooks 钩子其实就是在这个哈希表里面对应阶段增加新的函数。

要实现装饰器注册 hooks实例化一个全局 harness = HarnessMiddleware()，装饰器内部用这个全局实例去注册：

```python
HOOK_POINTS = [
    "on_session_start",
    "pre_think",
    "pre_tool_call",
    "post_tool_call",
    "post_reflect",
    "on_session_end",
]


class HarnessMiddleware:
    def __init__(self):
        self._hooks = defaultdict(list)

    def register(self, hook_point: str, name: str, fn, priority: int = 100):
        if hook_point not in HOOK_POINTS:
            raise ValueError(f"未知 hook_point: {hook_point}")

        self._hooks[hook_point].append((name, fn, priority))
        self._hooks[hook_point].sort(key=lambda item: item[2])

    async def run(self, hook_point: str, context: dict) -> dict:
        for name, fn, priority in self._hooks.get(hook_point, []):
            result = await fn(context)
            if result is not None:
                context = result

        return context


# 先实例化全局 middleware
harness = HarnessMiddleware()


# 再定义装饰器
def harness_hook(hook_point: str, name: str, priority: int = 100):
    def decorator(fn):
        harness.register(
            hook_point=hook_point,
            name=name,
            fn=fn,
            priority=priority,
        )
        return fn

    return decorator
```

调用方式，在 AgentLoop 里面显式调用，比如：
```python
ctx = await harness.run("on_session_start", {
    "query": query,
    "thread_id": thread_id,
    "user_id": user_id,
})
```

工具调用阶段也是同理，不过埋点位置在 ToolNode
```python
class HarnessToolNode(ToolNode):
    async def _run_one_tool(self, tool_call: dict, config: dict) -> dict:
        # 工具执行前：触发 pre_tool_call hooks
        ctx = await harness.run("pre_tool_call", {
            "tool_name": tool_call["name"],
            "tool_args": tool_call["args"],
            "tool_call_id": tool_call["id"],
        })

        if ctx.get("_rejected"):
            return {
                "tool_call_id": tool_call["id"],
                "content": f"[Harness 拒绝] {ctx['_reject_reason']}",
            }

        # 真正执行工具
        result = await super()._run_one_tool(tool_call, config)

        # 工具执行后：触发 post_tool_call hooks
        ctx = await harness.run("post_tool_call", {
            "tool_name": tool_call["name"],
            "tool_result": result["content"],
            "duration_ms": result.get("duration_ms", 0),
        })

        # post hook 可以修改工具结果
        if "tool_result" in ctx:
            result["content"] = ctx["tool_result"]

        return result
```


### Hooks

#### 1. on_session_start

Agent 开始工作前执行，用来初始化本次任务的运行上下文。

- **set_thread_context()**，初始化本次任务的 *ContextVar*，写入 *thread_id* 和 *session_dir*，让深层工具、AGUI/WebSocket 推送、子 Agent 能拿到当前任务上下文，避免多用户并发串台。
- **init_budget()**，初始化本次请求的 *TokenBudget*，记录 token 总预算、已消耗量、剩余额度和模型档位，后续用于预算检查、模型降级、压缩或提前收束。

#### 2. pre_think

LLM 推理前触发，主要用于在每轮 Think 前注入运行时提示。

- **TokenBudget hint 注入**，根据当前 token 预算状态给模型注入降级提示，比如进入 *minimal* 档时提醒模型“不要继续探索，基于已有 Observation 收笔”。
- **inject_runtime_messages**，统一注入运行时提示。它会根据当前 *TokenBudget* 状态生成降级 hint，比如进入 *minimal* 档时提醒模型“不要继续探索，基于已有 Observation 收笔”；同时也会把上一轮 **loop_detector / drift_check / budget_check** 写入的 *context.inject_messages* 合并到当前 messages，让模型在下一轮 Think 前看到纠偏、收笔或阶段约束提示。

#### 3. pre/post_tool_call

工具执行前后触发，用来治理工具调用边界。

pre_tool_call：

- **tool_whitelist**，工具白名单校验。最先执行，如果模型幻觉出未注册工具，直接拒绝。
- **phase_check**，阶段权限检查。根据当前 Agent 阶段判断这个工具能不能用，比如 *PLANNING* 阶段不能直接调用 *shopping_summary*。
- **schema_validate**，参数 schema 校验。检查 *tool_call["args"]* 是否符合工具参数定义，避免参数缺失、类型错误。

post_tool_call：

- **content_filter**，工具结果内容过滤，避免把敏感或不安全内容继续喂回模型。
- **truncate**，工具结果截断，防止单个工具返回过长导致上下文膨胀。
- **breaker_record**，记录工具调用结果，用于熔断统计，避免外部依赖异常时被反复调用。
- **step_assertion**，单步断言检查，校验当前工具结果是否满足预期格式或业务约束。

#### 4. post_reflect

ReAct Agent 每轮 Reflect 之后触发，用于做循环检测、目标偏移检查、token 余量检查和阶段转移。

- **loop_detector**，检测工具调用循环，比如连续多次调用同一个工具；触发后往下一轮注入“收笔/换思路”提示。
- **drift_check**，检测 Agent 行为是否偏离用户原始需求；如果轻微/严重漂移，就注入纠偏提示。
- **budget_check**，检查 token 预算余量，必要时触发压缩、降级或准备 fallback。
- **phase_transition**，根据当前执行结果推动阶段流转，比如 *PLANNING -> SEARCHING -> COMPARING -> CONCLUDING*。

补充讲一下 *drift_check* ，当然不是直接用一个 Lite 模型去校验，而是规则校验，如果疑似偏移就调用 Lite 模型去校验。也不是每轮都跑，而是 3 轮跑一次。

```python
def _computational_drift_check(query: str, actions: str, context: dict) -> str:
    """快速计算检查，返回 normal / suspicious。"""
    keywords = set(re.findall(r'\w+', query))
    action_words = set(re.findall(r'\w+', actions))
    overlap = keywords & action_words

    if len(overlap) / max(len(keywords), 1) < 0.2:
        return "suspicious"
    return "normal"
```

Lite 模型检测结果有三个 “正常” ,“轻量偏移”，“严重偏移”，“连续严重”。只有 “连续严重” 的时候会要求 LLM 基于已有观察，立即调用 ShoppingSummary 终止任务。

#### 5. on_session_end

AgentLoop 收束后、最终结果返回给前端前触发，通常发生在 *shopping_summary* 产出结果之后。

- **output_guard**，最终输出审核。对应 *audit_final_output() / audit_output()*，用于检查并处理最终回答里的风险内容，比如内部 *item_id*、API Key、敏感信息泄露等。
- **store_writeback**，长期记忆写回。对应 *writeback_preferences()*，把本轮对话中识别出的新用户偏好写回 Store，供下次会话使用。

其中 **store_writeback** 写回的是本轮识别出的新偏好，来源通常是 *shopping_summary.learned_preferences*，或者从最终 *trajectory / final message* 里取出对应字段。


## Agent Evolve

一个上线后不再迭代的 Agent，即便初期表现优异，也会因外部环境的动态变化而逐渐退化；这种退化并非源于代码本身的劣化，而是受限于

- 模型升级导致的 Prompt 行为漂移
- API 更新引发的工具不兼容
- 用户需求演化带来的新场景知识缺失
- 竞品提升所推高的用户期望水位

四类退化信号：

| 退化类型 | 表现 | 典型来源 | 监控指标 |
| :--- | :--- | :--- | :--- |
| **Quality Drift** | Rubric 日均分持续下滑 | • 闭源模型版本升级 <br> • 数据分布偏移 | Rubric 7日均分趋势 |
| **Tool Decay** | 某平台工具成功率骤降 | • 平台 API 行为变更 <br> • 反爬策略升级 | 工具成功率滑动窗口 |
| **Preference Stale** | 用户偏好变了但 Agent 还按旧偏好推 | • Store 里的偏好过期 <br> • 季节性变化 | 偏好命中率下降 |
| **Context Rot** | Prompt 和实际工具集不匹配 | • 新增/删除工具但 prompt 未同步 | 工具调用 404 率 |

退化通常不是断崖式的（那种容易发现），而是每天降 0.5%——连续一个月后从 79 分降到 68 分，但因为没人每天看分数曲线，等发现时已经积累了严重的用户体验损失。
自进化的核心动机：不能等人发现了再修，要让系统自己发现并自己修复。


### 业界 Agent 3×3 进化矩阵

#### What Evolves

> 工程原则：优先外层迭代，外层无法满足再向内层推进。外层便宜快速可逆；内层昂贵缓慢持久。

| 层级 | 内容 | 特点 |
| :--- | :--- | :--- |
| Layer1 External Files | 记忆、知识库、技能库、成功策略 | 低成本、即时生效、可回滚 |
| Layer2 Agent Harness | Prompt、工具定义、工作流、路由规则 | 成本中等 |
| Layer3 Model Weights | 参数知识、底层决策能力 | 成本高、迭代慢、效果持久 |

构建Agent系统时遵循**由外向内迭代**：
1. 优先更新外部知识库、记忆文件（Layer1）
2. 其次调整Prompt、工作流、工具链（Layer2）
3. 最后才考虑微调模型权重（Layer3）


#### When Persist

| 持久范围 | 作用域 | 案例 |
| :--- | :--- | :--- |
| Single Session | 单次会话内 | 临时 scratchpad、工具缓存 |
| Across Sessions | 同一用户、跨会话 | 用户偏好、技能库、会话Checkpoint |
| Across Users | 全体用户群体 | 模型微调、全局Prompt升级 |


#### 完整3×3矩阵

| | External Files | Agent Harness | Model Weights |
| :--- | :--- | :--- | :--- |
| **Single Session** | 临时记忆 scratchpad | 动态工具编排、会话状态机 | 极少使用：Test-time Training |
| **Across Sessions** | 用户偏好Store + 个人技能库 | Prompt版本管理、动态工具增删 | 参数化个性化（前沿方案） |
| **Across Users** | 共享知识库、公共策略库 | 全局默认Prompt迭代升级 | SFT / RL 模型训练 |


#### 现在已有的部分

| 矩阵格子 | Globex 已有 | 还缺 |
| :--- | :--- | :--- |
| Files × Session |  Store 跨会话持久化 | √ |
| Files × Across | Store 跨会话持久化 | **成功策略 / Skill Library** |
| Files × Users | CategoryInsight RAG 知识库 | **共享策略 Commons** |
| Harness × Session | 17-4 阶段状态机 / Token 预算 hint | √ |
| Harness × Across | 无 | **Prompt 版本化 + A/B（18-3）** |
| Harness × Users | 无 | **默认 prompt 自动升级（18-3）** |
| Weights × Session | $ \times $ | $ \times $（成本不合理） |
| Weights × Across | $ \times $ | $ \times $（Frontier 研究） |
| Weights × Users |  SFT + RL | **Bad case 飞轮持续供给** |


### Evolutionary Feedback Loop

#### MAPE 

借鉴 NVIDIA/IBM 的 MAPE (Monitor-Analyze-Plan-Execute) 数据飞轮框架。

{{< figure
    src="mape.png"
    caption="Fig. 1. MAPE。"
    align="center"
    width="90%"
>}}

基于 MAPE 理论，每个阶段做对应工程实现：

| 步骤 | 做什么 | 依赖什么基建 |
| :--- | :--- | :--- |
| Monitor | • 发现 Rubric 分下降 <br> • 工具失败率上升 <br> • 偏好命中率下降 | LangFuse Trace / Span / Score |
| Analyze | 按 P0/P1/P2 定位根因：<br> • 是格式问题还是决策问题 <br> • 知识缺失 | Rubrics as Rewards 评测体系 |
| Plan | 根据根因选择进化路径 <br> • 改记忆 <br> • 改 prompt <br> • 训模型） | 进化路径决策表 |
| Evolve | 执行进化并验证效果 | • Bad Case 数据飞轮 <br> • Prompt 自进化 <br> • 记忆进化 |

#### 进化路径选择

Monitor 发现退化后，不能一上来就训模型。训模型是最贵、最慢、最难回滚的手段，应该放到最后。

| 根因类型 | 优先路径 | 为什么 |
| :--- | :--- | :--- |
| 偏好过期 / 知识缺失 | 改记忆 / 改知识库 | 秒级生效，风险最低 |
| Prompt 规则缺失 | 改 Prompt | 分钟级生效，可以 A/B 验证 |
| 工具行为变化 | 改工具 / Harness Hook | 小时级修复，属于工程治理 |
| 模型决策质量不足 | SFT / RL | 成本最高，但能提升长期能力上限 |

简单判断流程：

```text
退化出现
  -> 是偏好 / 知识问题？优先改 Store / Skill
  -> 是规则缺失？改 Prompt 并做 A/B
  -> 是工具变化？改 Tool / Harness Hook
  -> 都不是，才进入 SFT / RL
```

这个原则可以概括为：**能改外层就不改内层，能改规则就不训模型。**

### Bad Case 数据飞轮

Bad Case 飞轮负责把线上失败样本变成可持续迭代的数据资产。它的入口通常是低分 Trace：

```text
LangFuse Score < 阈值
  -> 采集完整 trajectory
  -> 按 Rubric 明细分成 P0 / P1 / P2
  -> 不同等级走不同修复路径
```

采集的不是一条孤立回答，而是一整个诊断包：

| 字段 | 作用 |
| :--- | :--- |
| query | 复现用户原始需求 |
| trajectory | 保存完整 AgentLoop 轨迹 |
| rubric_score | 判断是否是 bad case |
| rubric_detail | 定位 P0 / P1 / P2 根因 |
| tool_calls | 分析工具调用顺序和参数 |
| token_consumed | 判断是否有成本失控 |
| trace_id | 回溯 LangFuse Trace |

核心采集逻辑可以很轻：

```python
COLLECTION_THRESHOLD = 0.65

async def should_collect(rubric_score: float) -> bool:
    return rubric_score < COLLECTION_THRESHOLD
```

采集后再按 P0/P1/P2 分流：

```python
def route_bad_case(rubric_detail: dict) -> str:
    if rubric_detail.get("p0_pass") is False:
        return "P0"

    if rubric_detail.get("p1_total_deduction", 0) >= 4:
        return "P1"

    if rubric_detail.get("p2_average", 5.0) < 3.0:
        return "P2"

    return "P2"
```

#### P0：秒级规则修复

P0 是红线问题，比如：

- 输出泄露内部 item_id
- 推荐违禁品
- 暴露工具名 / API Key
- 严重格式错误导致前端不可用

这类问题不需要训模型，因为它们通常是确定性可判断的。直接生成 Hook 规则，挂到 **output_guard / pre_tool_call / post_tool_call** 更快。

```python
async def auto_fix_p0(case):
    comment = case.rubric_detail.get("p0_comment", "")

    if "泄露" in comment:
        patterns = extract_leaked_patterns(case.trajectory)
        output_guard.add_patterns(patterns)

    if "违禁" in comment:
        blacklist.add(case.rubric_detail["bad_category"])
```

P0 修复不是临时 patch，而是持久化进 Harness 规则。后续所有请求都会经过这层防线。

#### P1：强模型重跑，进入 SFT

P1 是执行规范问题，比如：

- 没调 ShoppingSummary 就直接回答
- 工具调用顺序错
- Markdown 结构不完整
- 参数格式不符合约束

这类问题不能直接把错误轨迹丢进训练集，否则会以错训错。正确做法是：用强模型重跑同一条 query，生成一条更好的轨迹，再过门禁进入 SFT。

```python
async def replay_with_strong_model(case):
    result = await run_agent(
        query=case.query,
        model_override=get_judge_llm(),
    )

    new_score = await evaluate_trajectory(result["trajectory"])
    old_score = case.rubric_detail["total_score"]

    if new_score["total_score"] - old_score >= 15:
        return {
            "query": case.query,
            "trajectory": result["trajectory"],
            "score": new_score["total_score"],
            "source": "p1_replay",
        }

    return None
```

入库前还要过门禁：

| 门禁 | 要求 |
| :--- | :--- |
| Rubric 分 | 大于等于 75 |
| 格式正确率 | 100% |
| 相比原始轨迹 | 至少提升 15 分 |
| 轨迹长度 | 不超过 16K token |

P1 的产物是高质量 SFT 样本，用来让模型学会“正确的 AgentLoop 范式”。

#### P2：进入 RL Reward 池

P2 是质量问题，比如：

- 推荐和 query 不够贴合
- 覆盖了显式需求但没覆盖隐式需求
- 决策建议没有价值
- 场景洞察力弱

这类问题不一定存在唯一标准答案，SFT 能教格式，但很难教“哪个决策更好”。所以 P2 更适合作为 RL 信号：

```text
P2 低分轨迹
  + 线上高分轨迹
  -> 按 query pattern 配对
  -> 形成好坏对比
  -> 进入 RL reward 池
```

最终形成三个节奏：

| 周期 | 处理对象 | 产出 | 生效速度 |
| :--- | :--- | :--- | :--- |
| 日级 | P0 红线 | 新 Hook / 新规则 | 秒级 |
| 周级 | P1 规范 | SFT 数据 / 新 checkpoint | 周级 |
| 月级 | P2 质量 | RL reward 信号 / 新 checkpoint | 月级 |

### Prompt 自进化

Prompt 也会过期。工具变了、模型版本变了、用户 query 类型变了，都会让旧 prompt 的行为开始漂移。

如果靠人工直接改 prompt，会有几个问题：

- 不知道修了 A 会不会坏了 B
- 没有版本记录，回滚困难
- 多人修改容易冲突
- System Prompt 一变，Prompt Cache 可能大面积失效

所以 Prompt 不能“随手改”，而要版本化。

#### Prompt Version

每次 prompt 改动都存成一个版本：

```python
@dataclass
class PromptVersion:
    version: str
    content: str
    changelog: str
    author: str
    status: str = "draft"   # draft / testing / active / retired
    rubric_score: float | None = None
```

版本号按语义化管理：

| 类型 | 例子 | 处理方式 |
| :--- | :--- | :--- |
| major | 改 AgentLoop 结构 | 全量回归 |
| minor | 新增工具说明 / 新规则 | A/B 测试 |
| patch | 修错字 / 微调措辞 | 快速上线 |

#### A/B 测试

Prompt 自进化的核心不是“自动生成一段新 prompt”，而是**新 prompt 不能直接全量上线**。

```python
def get_prompt_for_user(user_id: str) -> str:
    if no_testing_version():
        return prompt_store.get_active().content

    ratio = hash(user_id) % 100 / 100.0
    if ratio < 0.10:
        return prompt_store.get_testing().content

    return prompt_store.get_active().content
```

A/B 期间看四个指标：

| 指标 | 判断 |
| :--- | :--- |
| Rubric 均分 | 新版本连续 3 天高于旧版本才放量 |
| 格式正确率 | 低于阈值直接回滚 |
| 工具失败率 | 明显升高则回滚 |
| Cache 命中率 | 大幅下降则告警 |

放量策略也要渐进：

```text
10% 流量观察
  -> 30% 流量观察
  -> 100% 全量
```

Prompt 更新还有一个容易忽略的点：**Prompt Cache 基于前缀匹配**。所以新规则尽量加在 prompt 末尾，不要频繁改开头的核心描述。A/B 测试期间也不要直接用 token 成本判定新 prompt 变差，因为新版本缓存还没预热，天然会更贵。

#### Auto Prompt Optimization

Prompt 自进化可以分三档：

| 阶段 | 做法 | 人工参与 |
| :--- | :--- | :--- |
| 手动 | 人看 bad case，手写规则 | 高 |
| 半自动 | LLM 分析 bad case，生成修改建议，人审核 | 中 |
| 全自动 | LLM 生成改动，自动 A/B，自动合入或回滚 | 低 |

项目里更稳的是半自动：让 judge LLM 读最近一批低分 bad case，输出修改建议，而不是直接改线上 prompt。

```python
async def suggest_prompt_improvement(bad_cases):
    current = prompt_store.get_active()
    summary = summarize_bad_cases(bad_cases)

    resp = await get_judge_llm().ainvoke([
        ("user", ANALYZE_PROMPT.format(
            current_prompt=current.content,
            bad_cases_summary=summary,
        ))
    ])

    return {
        "suggestion": resp.content,
        "based_on_version": current.version,
    }
```

自动化也要有门禁：

| 门禁 | 约束 |
| :--- | :--- |
| 修改幅度 | diff 不超过固定字数 |
| 工具描述 | 不能删除已有工具说明 |
| 核心流程 | 不能自动改 fork / 阶段规则 |
| 频率 | 每天最多生成一个新版本 |

原则是：**自动系统可以加规则、改措辞，但不能擅自改架构。**

### Memory & Strategy Evolution

记忆层进化是成本最低、反馈最快的一层。

最初的 Store 只存用户偏好：

```text
用户说“不要塑料”
  -> ShoppingSummary 提取 learned_preferences
  -> store_writeback 写入 Store
  -> 下次 on_session_start 读取并注入 prompt
```

但只存偏好还不够。升级后的 Store 还要存两类东西：

- **成功策略**：高分轨迹里可复用的做法
- **失败教训**：低分轨迹里应该避免的坑

#### 成功策略

成功策略不是完整轨迹，而是压缩后的经验：

```text
某类 query
  + 某种工具调用顺序
  + 高 Rubric 分
  -> 提炼成可复用策略
```

一条策略可以长这样：

```python
class StrategyEntry(BaseModel):
    strategy_id: str
    query_pattern: str
    summary: str
    tool_hints: list[str]
    key_decisions: list[str]
    rubric_score: float
    confidence: float = 1.0
    times_referenced: int = 0
```

例如：

```text
query_pattern: 跨平台比价类，带预算和材质约束
summary: 先确认品类属性，再并行搜索，最后综合比价和筛选
tool_hints:
  - category_insight
  - dispatch_tool
  - price_compare
  - shipping_calc
  - item_picker
  - shopping_summary
```

策略来自高分轨迹，但不是所有高分轨迹都能入库：

| 门禁 | 条件 |
| :--- | :--- |
| Rubric 分 | 大于等于 0.80 |
| 轨迹轮次 | 至少 4 轮 Act |
| 工具多样性 | 至少 3 个不同工具 |
| 去重 | 和已有策略不能太相似 |

新会话开始时，系统按当前 query 检索 1~3 条相似策略，注入 system prompt：

```python
def inject_strategies(prompt: str, strategies: list[StrategyEntry]) -> str:
    if not strategies:
        return prompt

    strategy_text = "\n".join(
        f"- {s.query_pattern}: {s.summary}"
        for s in strategies[:3]
    )

    return prompt + f"""
# 成功策略参考
{strategy_text}

注意：以上策略仅供参考，请根据当前 query 灵活调整。
"""
```

这里重点是“参考”而不是“强制”。策略库不是要把 Agent 变成固定流程，而是减少重复探索。

#### 失败教训

低分轨迹也有价值。它可以沉淀成“不要这么做”的经验：

```python
class LessonEntry(BaseModel):
    lesson_id: str
    query_pattern: str
    what_went_wrong: str
    avoid_hints: list[str]
    rubric_comment: str
    confidence: float = 1.0
```

注入时可以和成功策略放在一起：

```text
# 成功策略参考
- 跨平台比价类：先 CategoryInsight，再 fork 多平台搜索

# 失败教训参考
- 送礼类 query：不要只搜商品名，要考虑收礼人的年龄、性别、爱好
- 旅行套装类 query：不要跳过 CategoryInsight，否则容易漏组件
```

这类记忆让 Agent 不只是“记住用户喜欢什么”，还会逐渐记住“哪些做法容易失败”。

#### 策略生命周期

策略也会过期，所以需要 confidence：

```python
def compute_confidence(strategy):
    days_old = days_since(strategy.created_at)
    time_decay = exp_decay(days_old, half_life=60)

    reference_boost = min(strategy.times_referenced * 0.05, 0.5)

    return min(1.0, strategy.rubric_score * time_decay + reference_boost)
```

- 60 天没人引用，置信度减半
- 被高分请求引用，置信度回升
- 被低分请求引用，置信度下降

这样 Store 不是越存越乱，而是会自然沉淀出长期有效的策略。

### Dynamic Fork Evolution

Globex 里 fork 多平台搜索不是永远固定的。平台状态会变：

- 某平台最近 API 经常超时
- 某平台在某品类下结果质量低
- 新平台接入后需要加入候选

所以 fork 候选可以根据历史成功率动态调整：

```python
class PlatformSuccessTracker:
    def record(self, platform: str, success: bool):
        ...

    def get_success_rate(self, platform: str) -> float:
        ...

    def should_fork_to(self, platform: str, threshold: float = 0.3) -> bool:
        return self.get_success_rate(platform) >= threshold
```

生成候选平台：

```python
def get_fork_candidates() -> list[str]:
    candidates = []

    for platform in DEFAULT_PLATFORMS:
        if platform_tracker.should_fork_to(platform):
            candidates.append(platform)

    return candidates[:4]
```

然后在 pre_think 里注入提示：

```python
@harness_hook("pre_think", name="fork_platform_hint", priority=30)
async def inject_fork_hint(context: dict):
    candidates = get_fork_candidates()

    context.setdefault("inject_messages", []).append({
        "role": "system",
        "content": f"当前推荐 fork 的平台：{', '.join(candidates)}"
    })

    return context
```

这本质上是 Harness 层的自适应：不改模型、不改 prompt，只根据线上反馈动态改变 Agent 的行动边界。

### 最终闭环

到这里，Agent Evolve 不是一个单点功能，而是一套闭环：

```text
线上请求
  -> LangFuse 记录 Trace
  -> Rubric 生成 Score
  -> 低分进入 Bad Case 池
  -> P0/P1/P2 分流
  -> 改 Hook / 改 Prompt / 改 Store / 训模型
  -> A/B 或灰度验证
  -> 新版本上线
  -> 继续监控
```

从成本和生效速度看：

| 进化层 | 修复对象 | 生效速度 | 典型手段 |
| :--- | :--- | :--- | :--- |
| External Files | 偏好、知识、策略、教训 | 秒级 | Store / Skill / Strategy |
| Agent Harness | Prompt、工具、Hook、路由 | 分钟到小时级 | Prompt A/B / Harness 规则 |
| Model Weights | 决策能力和范式理解 | 周到月级 | SFT / RL |

所以 Globex 的自进化不是“模型自己训练自己”这么玄的东西，而是一个工程闭环：**先观测，再分级，再选择最低成本的修复路径，最后用 Rubric 和线上指标验证效果。**

越用越聪明，背后其实是三件事在持续发生：

1. **记忆层**记住用户偏好、成功策略和失败教训；
2. **Harness 层**通过 Prompt、Hook、工具路由持续修正行为边界；
3. **Weights 层**用高质量轨迹和 reward 信号慢慢提升决策上限。