---
title: "大语言模型对齐"
date: 2026-07-06T09:44:00+08:00
author: "dengkoicat"
tags: ["Deep Learning", "Reinforcement Learning", "LLM"]
categories: ["Deep Learning", "LLM"]
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

## 前言

大语言模型对齐（alignment）相关概念很多，例如 **SFT**、**RLHF**、**PPO**、**DPO**、**GRPO**、**GSPO** 等。它们并不是同一层级的东西，可以先按“训练阶段”和“优化方法”来理解：

```text
预训练 Base Model
        ↓
SFT：监督微调，让模型学会按指令回答
        ↓
偏好 / 强化对齐阶段
        ├── RLHF：先训练奖励模型，再用强化学习优化模型
        │       └── PPO：经典 RLHF 优化算法
        │
        ├── DPO：不显式训练奖励模型，也不跑在线 RL，直接用偏好数据优化模型
        ├── GRPO：去掉 Critic 的组相对策略优化，常用于推理增强
        └── GSPO：序列级策略优化，提升大模型 RL 训练稳定性
```

简单来说，SFT 解决“模型会不会按指令回答”，后续的偏好对齐解决“模型的回答是否更符合人类偏好或任务奖励”。

本文只展开 **PPO 在 RLHF 中的作用**。DPO、GRPO、GSPO 这里仅作为定位：它们都属于偏好对齐或强化对齐路线，但训练目标、数据使用方式、是否需要 Reward Model、是否需要在线采样都不同。

{{<figure
    src="aligned-against-baseline.png"
    caption="Fig. 1. 在 InstructGPT 实验中，SFT 和 PPO 训练后的模型都优于 GPT(Prompt) 基线。Image source: Training language models to follow instructions with human feedback, OpenAI, 2022."
    align="center"
    width="90%"
>}}

## SFT

SFT（Supervised Fine-Tuning）是 LLM 对齐训练的第一步，本质上是有监督训练。经过 Pretrain 的 LLM 只是一个会预测下一个 token 的 Base Model；SFT 的目的就是让这个 Base Model 变成一个可以**遵循指令**的 Instruct Model。

首先，我们需要整理高质量的指令数据集。每条样本通常包含用户输入和理想回答，这就是 SFT 的监督源。模型在训练过程中会学习“在这种指令下应该如何回答”，包括回答格式、语气、领域规范和安全边界。

SFT 和 Pretrain 的核心都是 next-token cross entropy，但二者有区别：

1. Pretrain 面向大规模通用语料，目标是学习语言、知识和模式；SFT 面向指令-回答数据，目标是学习交互方式。
2. Pretrain 通常对每个 token 算 loss；SFT 通常只对 assistant 的回答部分算 loss，用户提问部分只作为条件上下文。
3. Pretrain 常是大规模全参训练；SFT 在工程中可以全参微调，也可以使用 LoRA / QLoRA 等参数高效微调方法。

### SFT 数据集

训练数据本质是一个 $(x,y)$，其中 $x$ 是输入指令，$y$ 是理想输出。多轮对话数据通常写成 conversation 格式：

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "你背后的模型是哪个版本？它由谁开发？"
    },
    {
      "role": "assistant",
      "content": "我是由 jingyaogong 开发的高效小参数 AI 模型。"
    },
    {
      "role": "user",
      "content": "你模型的训练数据来源是什么？"
    },
    {
      "role": "assistant",
      "content": "我的训练数据涵盖多领域，确保覆盖广泛，但具体细节不公开。"
    }
  ]
}
```

如果把用户输入记为 $x$，assistant 回答记为 $y=(y_1,\dots,y_T)$，SFT 的损失可以写成：

$$
\mathcal{L}_{\text{SFT}}(\theta) =-\sum_{t=1}^{T}\log \pi_\theta(y_t \mid x, y_{\lt t})
$$

更贴近工程实现的写法会加入 mask，只在 assistant token 上计算 loss：

$$
\mathcal{L}_{\text{SFT}}(\theta) =-\sum_{t=1}^{T} m_t \log \pi_\theta(y_t \mid y_{\lt t})
$$

其中 $m_t=1$ 表示该 token 属于 assistant 回答，需要计入损失；$m_t=0$ 表示该 token 属于 prompt、system message 或其他不需要监督的位置。

### SFT 优缺点

SFT 的优势：

- 训练稳定，成本相对低，少量高质量样本就能明显改善模型行为。
- 能提升指令遵循、对话格式、回答风格和领域规范。
- 数据和训练流程直观，便于调试和复现。

SFT 的缺点：

- 训练结果高度依赖数据集质量，低质量样本会让模型学到坏格式、坏习惯甚至错误知识。
- 本质是模仿标准答案，不能直接表达“两个回答哪个更好”的偏好信息。
- 如果学习率、训练轮数或数据分布控制不好，可能破坏 Base Model 原有能力，出现灾难性遗忘。
- 对安全、伦理、复杂推理和多目标权衡的提升有限，因为这些问题往往不是单一标准答案能完全覆盖的。

## PPO，近端策略优化

已经有一个会回答问题的模型，如何通过人类偏好继续优化它，让它生成更符合人类意图的答案？这就是 RLHF（Reinforcement Learning from Human Feedback）的目标。

比如 LLM 知道 `1+1=2`，也知道“一加一等于二”。如果某个应用场景更偏好简洁、符号化的答案，SFT 只能让模型模仿训练集中出现的写法，却很难稳定表达“这个回答比另一个回答更好”。偏好对齐要解决的正是这个问题。

典型 RLHF 可以分成三步：

1. 先训练一个经过 SFT 的初始策略模型 $\pi_{\text{SFT}}$。
2. 收集同一 prompt 下多个回答的人类偏好排序，训练奖励模型 $r_\phi(x,y)$。
3. 用 PPO 微调语言模型，让它在不偏离参考模型太远的前提下获得更高奖励。

PPO 在这里不是“让模型学标准答案”，而是把语言模型看成一个策略（Policy），让它通过奖励信号调整输出分布。

### Reward Model 与 Bradley-Terry

RLHF 中的 Reward Model（RM）负责给模型回答打分。它不是直接从“正确答案”里学出来的，而是从人类偏好比较中学出来的。

假设同一个 prompt $x$ 下有两个回答：

$$
y_w \succ y_l
$$

其中 $y_w$ 是人类更喜欢的回答，$y_l$ 是人类不太喜欢的回答。Reward Model 给两个回答打分：

$$
r_\phi(x,y_w),\quad r_\phi(x,y_l)
$$

Bradley-Terry 模型是一种经典的成对比较概率模型。它关心的不是两个回答的绝对分数，而是分数差：

$$
P(y_w \succ y_l|x) = \sigma(r_\phi(x,y_w)-r_\phi(x,y_l))
$$

对应损失为：

$$
\mathcal{L}_{RM}(\phi) = -\log\sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right)
$$

这个 loss 的直觉很简单：让偏好回答 $y_w$ 的分数高于 rejected 回答 $y_l$。训练完成后，Reward Model 通常会被冻结，作为 PPO 阶段的奖励函数近似。

需要注意：Reward Model 学到的是人类偏好的近似，不是真理本身。如果偏好数据有偏、标注标准不稳定，或者 Reward Model 泛化不好，后续 PPO 就可能放大这些问题。

### LLM 如何对应强化学习

在 RLHF 中，LLM 可以被建模成一个策略：

| 强化学习概念 | LLM 中的对应物 |
| :--- | :--- |
| 状态 $s_t$ | prompt 加上当前已经生成的 token，即 $(x,y_{\lt t})$ |
| 动作 $a_t$ | 下一个 token $y_t$ |
| 策略 $\pi_\theta(a_t \mid s_t)$ | 当前 LLM 的 token 概率分布 |
| 轨迹 $\tau$ | 一次完整生成 $(x,y)$ |
| 奖励 $r_\phi(x,y)$ | Reward Model 对完整回答的评分 |

给定 prompt $x$，LLM 按照自回归方式生成回答：

$$
y=(y_1,\dots,y_T)
$$

整段回答的概率可以写成：

$$
\pi_\theta(y|x) = \prod_{t=1}^{T} \pi_\theta(y_t|x,y_{\lt t})
$$

如果只看奖励最大化，目标可以写成：

$$
\max_\theta \mathbb{E}_{y\sim \pi_\theta(\cdot|x)} \left[ r_\phi(x,y) \right]
$$

但直接最大化 Reward Model 分数会出问题。模型可能钻 RM 的空子，生成高分但怪异、啰嗦、谄媚、重复或分布崩坏的回答。这就是 reward hacking。

因此，LLM 的 PPO 通常不是只最大化 RM 分数，而是最大化带 KL 约束的目标：

$$
\max_\theta \mathbb{E}_{y\sim \pi_\theta} \left[ r_\phi(x,y) - \beta D_{\mathrm{KL}} \left( \pi_\theta(\cdot \mid x) \| \pi_{\mathrm{ref}}(\cdot \mid x) \right) \right]
$$

其中：

- $\pi_{\mathrm{ref}}$ 通常是 SFT 后冻结的参考模型。
- $r_\phi(x,y)$ 是 Reward Model 给完整回答的分数。
- $\beta$ 控制 KL 约束强度。$\beta$ 越小，模型越追求奖励，变化越大；$\beta$ 越大，模型越保守。

这也是 RLHF 中 PPO 的核心矛盾：既要让模型变得更符合偏好，又不能让它偏离原来的语言能力和指令遵循能力太远。

### KL 惩罚如何落到 token 上

完整计算两个语言模型分布之间的 KL 很贵，因为每一步都涉及整个词表分布。工程上常用 sampled token 上的 log-prob difference 作为近似 KL 惩罚。

生成第 $t$ 个 token 后，可以写成：

$$
r_t^{KL} = -\beta \left[ \log \pi_\theta(y_t|x,y_{\lt t}) - \log \pi_{\mathrm{ref}}(y_t|x,y_{\lt t}) \right]
$$

如果当前模型比参考模型更强烈地倾向这个 token，括号内为正，KL 惩罚为负；如果当前模型没有明显偏离参考模型，惩罚较小。

最终 token-level reward 常见形式是：

$$
r_t = \begin{cases} r_t^{KL}, & t \lt T \\ r_\phi(x,y)+r_t^{KL}, & t=T \end{cases}
$$

也就是说，中间 token 主要承受 KL 惩罚；回答结束时，再把 Reward Model 对完整回答的分数加到最后一步。

更直观地看，一条回答的总奖励大致是：

$$
R(x,y) = r_\phi(x,y) - \beta \sum_{t=1}^{T} \left[ \log \pi_\theta(y_t|x,y_{\lt t}) - \log \pi_{\mathrm{ref}}(y_t|x,y_{\lt t}) \right]
$$

这里要纠正一个常见误解：KL 不是“奖励模型给的奖励”，而是约束项。它的作用是防止当前策略为了追求 RM 高分而偏离参考模型太远。

### Value Model

PPO 是 Actor-Critic 风格的方法。Actor 是当前要优化的语言模型策略 $\pi_\theta$；Critic 是 Value Model，用来估计当前状态未来能拿到多少回报。

Value Model 预测：

$$
V_\psi(s_t) \approx \mathbb{E} \left[ \sum_{k=0}^{T-t}\gamma^k r_{t+k} \mid s_t \right]
$$

其中 $s_t=(x,y_{\lt t})$。对于 LLM 来说，状态不是图像或棋盘，而是“prompt + 当前已经生成的 token 序列”。

工程上，Value Model 通常不会从随机权重开始训练，而是在语言模型 backbone 后面接一个 Value Head：

- Actor Head 输出词表分布，可以理解为 $d_{\text{model}} \times |V|$。
- Value Head 输出一个标量，可以理解为 $d_{\text{model}} \times 1$。

拿到一条轨迹的奖励序列 $[r_1,r_2,\dots,r_T]$ 后，可以计算每个状态的经验回报：

$$
G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}
$$

最直接的 Value Loss 是均方误差：

$$
L_t^{\text{unclip}}(\psi) = \left( V_\psi(s_t)-G_t \right)^2
$$

很多 PPO 实现会使用 value clipping。记录更新前的旧预测值 $V_{\text{old}}(s_t)$，限制新 value 不要一次变化太大：

$$
V_{\psi}^{\text{clip}}(s_t) = V_{\text{old}}(s_t) + \text{clip} \left( V_\psi(s_t)-V_{\text{old}}(s_t), -\epsilon, \epsilon \right)
$$

对应 clipped value loss：

$$
L_t^{\text{clip}}(\psi) = \left( V_{\psi}^{\text{clip}}(s_t)-G_t \right)^2
$$

最终 Value Loss 常写成：

$$
L_{V}(\psi) = \frac{1}{2} \max \left( L_t^{\text{unclip}}(\psi), L_t^{\text{clip}}(\psi) \right)
$$

这里用 max 是一种保守策略：如果 clipped 和 unclipped 两个版本中有一个误差更大，就按更大的误差惩罚，避免 value function 更新过猛。



### Advantage

只知道 reward 还不够。PPO 真正更新策略时，关心的是某个动作比当前状态下的平均水平好多少，这就是 Advantage。

最简单的优势函数可以写成：

$$
A_t = G_t - V_\psi(s_t)
$$

如果 $A_t>0$，说明这个 token 选择带来的回报高于 Value Model 的预期，应该提高它的概率；如果 $A_t<0$，说明这个 token 选择低于预期，应该降低它的概率。

实际 PPO 中常用 GAE（Generalized Advantage Estimation）来降低方差。先定义 TD residual：

$$
\delta_t = r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t)
$$

再计算：

$$
A_t^{GAE} = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}
$$

其中：

- $\gamma$ 是折扣因子。
- $\lambda$ 控制 bias-variance trade-off。
- $\lambda$ 越接近 1，越接近完整 Monte Carlo return，偏差小但方差大。
- $\lambda$ 越接近 0，越依赖一步 TD，方差小但偏差大。

在 LLM RLHF 中，奖励通常比较稀疏：RM 分数主要在回答结束时给出，中间 token 主要是 KL 惩罚。因此 Advantage 的估计质量会直接影响训练稳定性。

### PPO-Clip

PPO 的核心是限制新旧策略之间的变化幅度。先定义重要性采样比率：

$$
\rho_t(\theta) = \frac{ \pi_\theta(y_t|s_t) }{ \pi_{\theta_{\text{old}}}(y_t|s_t) }
$$

如果 $\rho_t(\theta)>1$，说明新策略比旧策略更倾向于生成这个 token；如果 $\rho_t(\theta)<1$，说明新策略降低了这个 token 的概率。

普通 policy gradient 会直接最大化：

$$
\rho_t(\theta)A_t
$$

但这样可能导致策略一步更新太大。PPO-Clip 的目标函数为：

$$
L_t^{\text{PPO}}(\theta) = \min \left( \rho_t(\theta)A_t,\, \text{clip} \left( \rho_t(\theta), 1-\epsilon, 1+\epsilon \right) A_t \right)
$$

clip 的作用要分情况理解：

- 如果 $A_t>0$，说明这个 token 比预期好，模型应该提高它的概率；但 $\rho_t$ 最多提高到 $1+\epsilon$，超过就不再给额外收益。
- 如果 $A_t<0$，说明这个 token 比预期差，模型应该降低它的概率；但 $\rho_t$ 最多降低到 $1-\epsilon$，超过也不再给额外收益。

这就是 PPO 里的 proximal：每次更新只允许策略在旧策略附近移动，避免一次梯度更新把模型推崩。

### PPO 的完整 Loss

实际训练中，PPO 往往把 policy loss、value loss、entropy bonus、KL penalty 组合起来。若按“最小化 loss”的写法，可以写成：

$$
L(\theta,\psi) = - \mathbb{E} \left[ L_t^{\text{PPO}}(\theta) \right] + c_v L_V(\psi) - c_e H(\pi_\theta) + \beta D_{\mathrm{KL}} \left( \pi_\theta \| \pi_{\mathrm{ref}} \right)
$$

其中：

- 第一项是 policy loss，用来提高高 Advantage token 的概率，降低低 Advantage token 的概率。
- 第二项是 value loss，用来训练 Critic。
- 第三项是 entropy bonus，用来鼓励探索，避免策略过早变得过于确定。
- 第四项是 KL penalty，用来约束当前模型不要偏离参考模型太远。

{{<figure
    src="ppo.png"
    caption="Fig. 2. PPO 过程"
    align="center"
    width="90%"
>}}


## DPO，直接偏好优化

前面讲 PPO 时，我们默认 RLHF 要走三步：

1. 先用 SFT 得到一个会说话的模型 $\pi_{\text{ref}}$；
2. 再用人类偏好数据训练一个奖励模型 $r_\phi(x,y)$；
3. 最后用 PPO 优化策略 $\pi_\theta$，让模型拿更高 reward，同时别偏离参考模型太远。

这个流程很自然，但也很“重”。PPO 本质上是在做强化学习：要采样、估计 advantage、调 KL penalty、控制 reward hacking，还要小心策略更新太猛导致模型崩掉。更麻烦的是，PPO 优化的 reward 不是人类偏好本身，而是一个先训练出来的 Reward Model。也就是说，整个链路里有两层误差：Reward Model 可能学歪，PPO 也可能把这个歪 reward 优化得更歪。

DPO 的漂亮之处在于：它问了一个更直接的问题。

既然我们最终想要的只是“chosen response 的概率相对 rejected response 更高”，为什么一定要显式训练 Reward Model，再跑一轮 RL？

DPO 的核心思想是：直接从偏好数据中优化语言模型，而不是显式训练 Reward Model，也不再使用 PPO 进行在线强化学习。

偏好数据仍然是成对形式：

$$
(x, y_w, y_l)
$$

其中：

- $x$ 是用户输入 prompt；
- $y_w$ 是人类更偏好的回答，也叫 chosen response；
- $y_l$ 是人类不偏好的回答，也叫 rejected response。

DPO 的目标不是让模型简单模仿 $y_w$，而是让当前策略模型 $\pi_\theta$ 相对于参考模型 $\pi_{\text{ref}}$，更加偏向 $y_w$，同时远离 $y_l$。

### 从 RLHF 到 DPO

前面 PPO 的 KL 约束目标可以写成：

$$
\max_{\pi_\theta} \mathbb{E}_{y \sim \pi_\theta(\cdot|x)} \left[ r(x,y) - \beta D_{\mathrm{KL}}\left( \pi_\theta(y|x) \| \pi_{\text{ref}}(y|x) \right) \right]
$$

这个目标的直觉是：既要让回答获得更高 reward，又不能让当前模型偏离参考模型太远。

DPO 的关键推导是：在这个 KL 约束优化问题下，最优策略 $\pi^*$ 和 reward 之间存在如下关系：

$$
r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

其中 $Z(x)$ 是归一化项，只和 prompt $x$ 有关。

这个式子的含义是：一个回答的 reward 越高，最优策略相对于参考模型就越应该提高这个回答的概率。

换句话说，Reward Model 可以被策略模型和参考模型之间的 log-prob 差值隐式表示出来。

这就是 DPO 的核心：不用单独训练 $r_\phi(x,y)$，而是直接用 $\pi_\theta$ 和 $\pi_{\text{ref}}$ 的概率差来表达偏好。

### DPO 损失函数

偏好建模仍然可以使用 Bradley-Terry 形式：

$$
P(y_w \succ y_l|x) = \sigma\left( r(x,y_w)-r(x,y_l) \right)
$$

将前面的 reward 表达式代入后，$Z(x)$ 会被抵消，得到：

$$
P(y_w \succ y_l|x) = \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right)
$$

因此 DPO 的训练损失可以写成：

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right)
$$

- $\beta$ 较大：模型对 chosen / rejected 的区分会更强，更新更激进。
- $\beta$ 较小：模型更新更保守，更接近参考模型

也可以展开成更直观的形式：

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\log \sigma\left( \beta \left[ \log \pi_\theta(y_w|x) - \log \pi_{\text{ref}}(y_w|x) \right] - \beta \left[ \log \pi_\theta(y_l|x) - \log \pi_{\text{ref}}(y_l|x) \right] \right)
$$

这里的 $\pi_{\text{ref}}$ 通常是 SFT 后冻结的模型，$\pi_\theta$ 是当前要训练的模型。

DPO 虽然简单稳定，但也不是万能的。

首先，DPO 强依赖偏好数据质量。如果 chosen / rejected 标注不稳定，或者数据里存在大量噪声，模型会直接学到这些偏差。

其次，DPO 是离线训练方法。它只利用已有偏好对，不像 PPO 那样可以让当前模型在线生成新回答，再根据 reward 继续探索。因此在某些需要主动探索的任务上，DPO 的能力可能受限。

最后，DPO 主要解决的是“两个回答哪个更好”的偏好学习问题。如果任务需要复杂的长期规划、多步工具调用、环境交互或可验证奖励，纯 DPO 可能不如在线 RL 方法灵活。


{{<figure
    src="ppo.png"
    caption="Fig. 3. DPO 过程"
    align="center"
    width="90%"
>}}

## GRPO，组相对策略优化

PPO 和 DPO 分别代表了两种思路：

- PPO：显式训练 Reward Model，在线采样回答，再通过 Value Model、Advantage 和 PPO-Clip 更新策略。
- DPO：不显式训练 Reward Model，也不在线采样，而是直接用离线偏好对优化 chosen / rejected 的相对概率。

GRPO（Group Relative Policy Optimization，组相对策略优化）更接近 PPO。它仍然是在线强化学习方法，也需要模型针对 prompt 生成回答，再根据奖励信号更新策略。

但 GRPO 对 PPO 做了一个关键简化：**去掉 Critic / Value Model**。

PPO 需要额外训练一个 Value Model 来估计状态价值 $V_\psi(s_t)$，再用它计算 Advantage。这个 Value Model 往往和策略模型规模接近，会带来明显的显存和计算开销。GRPO 的做法是：不再训练单独的 Value Model，而是在同一个 prompt 下采样多条回答，用这组回答的奖励均值作为 baseline。

所以 GRPO 的核心可以概括为：

- PPO：用 Value Model 估计“这个状态平均能拿多少奖励”；
- GRPO：用同一 prompt 下多条回答的组内平均奖励，估计“这一组回答的平均水平”。

### 从 PPO 到 GRPO

PPO 中，Advantage 通常写成：

$$
A_t = G_t - V_\psi(s_t)
$$

其中 $V_\psi(s_t)$ 来自 Value Model。它的作用是提供一个 baseline：如果当前动作带来的回报高于 baseline，就提高这个动作的概率；如果低于 baseline，就降低它的概率。

GRPO 不再训练 $V_\psi(s_t)$。对于同一个 prompt $x$，GRPO 从旧策略模型 $\pi_{\theta_{\text{old}}}$ 中采样一组回答：

$$
\{y_1,y_2,\dots,y_G\} \sim \pi_{\theta_{\text{old}}}(\cdot|x)
$$

然后用 Reward Model 或规则奖励函数对每个回答打分：

$$
r_i = r(x,y_i), \quad i=1,\dots,G
$$

这时不需要问“某个 token 的状态价值是多少”，而是直接比较同一个 prompt 下不同回答的相对好坏。

如果某个回答的奖励高于组内平均水平，它就是相对好的回答；如果低于组内平均水平，它就是相对差的回答。

这就是 Group Relative 的含义。

### 组相对 Advantage

GRPO 的 Advantage 通常按组内奖励归一化计算：

$$
A_i = \frac{r_i - \text{mean}(r_1,\dots,r_G)}{\text{std}(r_1,\dots,r_G)}
$$

其中：

- $r_i$ 是第 $i$ 个回答的奖励；
- $\text{mean}(r_1,\dots,r_G)$ 是同一 prompt 下这组回答的平均奖励；
- $\text{std}(r_1,\dots,r_G)$ 是这组奖励的标准差；
- $A_i$ 表示第 $i$ 个回答相对于同组其他回答的好坏。

如果 $A_i \gt 0$，说明 $y_i$ 比这组回答的平均水平更好，训练时应该提高它的生成概率。

如果 $A_i \lt 0$，说明 $y_i$ 比这组回答的平均水平更差，训练时应该降低它的生成概率。

和 PPO 不同的是，GRPO 的 Advantage 不依赖 Value Model，而是直接来自组内奖励比较。

这对推理任务很自然。比如同一道数学题，模型一次生成 8 个答案，其中有的推理正确，有的推理错误。GRPO 不需要额外判断每个中间状态的价值，只需要比较这 8 个答案的最终奖励，就能得到相对优势。

### GRPO 目标函数

GRPO 仍然保留了 PPO-Clip 的思想，用新旧策略概率比限制每次更新的幅度。

先定义第 $i$ 个回答中第 $t$ 个 token 的概率比：

$$
\rho_{i,t}(\theta) = \frac{\pi_\theta(y_{i,t}|x,y_{i,\lt t})}{\pi_{\theta_{\text{old}}}(y_{i,t}|x,y_{i,\lt t})}
$$

GRPO 的策略优化目标可以写成：

$$
\mathcal{L}_{\text{GRPO}}(\theta) = - \frac{1}{G}\sum_{i=1}^{G}\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\left[\min\left(\rho_{i,t}(\theta)A_i,\text{clip}(\rho_{i,t}(\theta),1-\epsilon,1+\epsilon)A_i\right)-\beta D_{\mathrm{KL}}(\pi_\theta \| \pi_{\text{ref}})\right]
$$

这个式子可以拆成三部分理解。

第一部分是概率比 $\rho_{i,t}(\theta)$，表示新策略相对于旧策略，是提高了还是降低了当前 token 的生成概率。

第二部分是组相对 Advantage $A_i$，表示当前回答在同组回答里是好还是差。

第三部分是 KL 约束，用来防止当前策略 $\pi_\theta$ 偏离参考模型 $\pi_{\text{ref}}$ 太远。

如果 $A_i \gt 0$，说明这个回答比组内平均更好，GRPO 会提高它的 token 概率；如果 $A_i \lt 0$，说明这个回答比组内平均更差，GRPO 会降低它的 token 概率。

和 PPO 一样，clip 的作用是防止策略更新过猛。即使某个回答奖励很高，模型也不能一次性把它的概率拉得太大；即使某个回答奖励很低，也不能一次性把它的概率压得太狠。

{{<figure
    src="grpo.png"
    caption="Fig. 4. GRPO 过程"
    align="center"
    width="90%"
>}}



## GSPO，序列级策略优化

前面讲 GRPO 时，我们说它相比 PPO 去掉了 Critic / Value Model，改用同一个 prompt 下多条回答的组内相对奖励来估计 Advantage。

但是 GRPO 仍然有一个问题：它的奖励通常是 sequence-level 的，也就是对整段回答 $y_i$ 打分；但它的 policy ratio 却是 token-level 的，也就是每个 token 都有一个独立的 $\rho_{i,t}(\theta)$。

这会带来一个粒度不一致的问题：

- reward 是整段回答级别的；
- advantage 是整段回答级别的；
- 但重要性采样比率和 clipping 是 token 级别的。

GSPO（Group Sequence Policy Optimization，组序列策略优化）就是针对这个问题提出的。它的核心思想是：**既然奖励是给整段回答的，那策略优化也应该尽量在序列级别完成。**

所以 GSPO 可以理解为 GRPO 的进一步改造：

- GRPO：组内相对 Advantage + token-level ratio；
- GSPO：组内相对 Advantage + sequence-level ratio。

### 从 GRPO 到 GSPO

GRPO 中，第 $i$ 个回答的第 $t$ 个 token 的概率比为：

$$
\rho_{i,t}(\theta) = \frac{\pi_\theta(y_{i,t}|x,y_{i,\lt t})}{\pi_{\theta_{\text{old}}}(y_{i,t}|x,y_{i,\lt t})}
$$

这个 $\rho_{i,t}(\theta)$ 表示：当前模型相比旧模型，对某个 token 的生成概率提高了还是降低了。

但问题是，GRPO 的 Advantage 通常是回答级别的：

$$
A_i = \frac{r_i - \text{mean}(r_1,\dots,r_G)}{\text{std}(r_1,\dots,r_G)}
$$

也就是说，同一个回答里的所有 token 共享同一个 $A_i$。

这样就会出现一个现象：明明 reward 是对整段回答的评价，但每个 token 却被单独计算 ratio 和 clip。某些 token 的概率波动可能会放大梯度噪声，导致训练不稳定。

GSPO 的改法很直接：不再给每个 token 单独算一个 ratio，而是给整段回答算一个 sequence-level ratio。

### 序列级重要性比

对第 $i$ 个回答 $y_i$，GSPO 定义序列级重要性比：

$$
s_i(\theta) = \left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{\text{old}}}(y_i|x)}\right)^{\frac{1}{|y_i|}}
$$

因为语言模型是自回归生成的，整段回答概率可以拆成每个 token 概率的乘积：

$$
\pi_\theta(y_i|x) = \prod_{t=1}^{|y_i|}\pi_\theta(y_{i,t}|x,y_{i,\lt t})
$$

所以 $s_i(\theta)$ 也可以写成：

$$
s_i(\theta) = \exp\left(\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\log \frac{\pi_\theta(y_{i,t}|x,y_{i,\lt t})}{\pi_{\theta_{\text{old}}}(y_{i,t}|x,y_{i,\lt t})}\right)
$$

这里的 $\frac{1}{|y_i|}$ 是长度归一化，作用是避免长回答因为 token 更多，导致概率乘积过小或 log-ratio 累积过大。

直观理解：

- GRPO 看的是“每个 token 的概率变化”；
- GSPO 看的是“整段回答平均意义上的概率变化”。

所以 GSPO 的优化单位更接近 reward 的单位。

### GSPO 目标函数

GSPO 保留了 GRPO 的组相对 Advantage，也保留了 PPO / GRPO 中的 clip 思想，但把 token-level ratio 换成了 sequence-level ratio。

如果按最大化目标写，可以写成：

$$
\mathcal{J}_{\text{GSPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\min\left(s_i(\theta)A_i,\text{clip}(s_i(\theta),1-\epsilon,1+\epsilon)A_i\right)\right]
$$

如果按最小化 loss 写，就是：

$$
\mathcal{L}_{\text{GSPO}}(\theta) = -\frac{1}{G}\sum_{i=1}^{G}\min\left(s_i(\theta)A_i,\text{clip}(s_i(\theta),1-\epsilon,1+\epsilon)A_i\right)
$$

其中：

- $G$ 是同一个 prompt 下采样的回答数量；
- $A_i$ 是第 $i$ 个回答的组相对 Advantage；
- $s_i(\theta)$ 是第 $i$ 个回答的序列级重要性比；
- $\epsilon$ 是 clip 范围，用来限制策略更新幅度。

如果 $A_i \gt 0$，说明这个回答比组内平均水平更好，GSPO 会提高整段回答的生成概率。

如果 $A_i \lt 0$，说明这个回答比组内平均水平更差，GSPO 会降低整段回答的生成概率。

和 GRPO 一样，GSPO 也可以结合 KL 约束或把 KL 惩罚放进 reward 里，用来防止当前模型偏离参考模型太远。只是 GSPO 的核心区别不在 KL，而在于把重要性比和 clipping 从 token 级别改成 sequence 级别。


GRPO 的核心是去掉 Value Model，用组内相对奖励估计 Advantage，是在 GRPO 的基础上进一步把策略更新从 token-level 改成 sequence-level。

GSPO 的优势主要有三点，
1. 训练信号更一致。因为很多 LLM 强化学习任务的 reward 本来就是给整段回答的，例如数学题答案是否正确、代码是否通过测试、推理链是否有效。GSPO 让优化粒度和奖励粒度对齐，减少 token-level ratio 带来的噪声。
2. 训练稳定性更好。GRPO 中每个 token 都有自己的 ratio，某些 token 的概率波动可能导致梯度不稳定。GSPO 使用整段回答的平均 ratio，可以减少这种细粒度噪声。
3. 对 MoE 模型更友好。MoE 模型中不同 token 可能激活不同专家，token-level ratio 对专家路由变化更敏感；GSPO 更关注整段回答的 sequence likelihood，因此对 token 级别的概率和路由波动不那么敏感。

当然，GSPO 也有代价。它把 ratio 放到序列级别后，牺牲了一部分 token-level 的细粒度控制。如果某个回答整体奖励不错，但其中某些 token 实际上很差，GSPO 不会像 token-level 方法那样精细地区分这些 token。


{{<figure
    src="gspo.png"
    caption="Fig. 5. GSPO 过程"
    align="center"
    width="90%"
>}}


## 总结

| 方法 | 核心思想 | 数据来源 | 是否在线采样 | 是否需要 Reward Model / 奖励函数 | 是否需要 Value Model / Critic | Advantage 计算 | Ratio / Clip 粒度 | KL / 约束方式 | 适合场景 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| PPO | 先训练 Reward Model，再用强化学习优化策略 | prompt + 在线生成回答 | 需要 | 需要 Reward Model | 需要 | $A_t = G_t - V_\psi(s_t)$，常配合 GAE | token-level ratio：$\rho_t(\theta)$；token-level clip | 通常显式加入 reference KL，防止偏离 $\pi_{\text{ref}}$ | 通用 RLHF，对齐能力强，但训练复杂 |
| GRPO | 去掉 Critic，用同一 prompt 下多条回答的组内相对奖励估计 Advantage | prompt + 在线生成多条回答 | 需要 | 需要 Reward Model 或规则奖励函数 | 不需要 | $A_i = \frac{r_i-\text{mean}(r)}{\text{std}(r)}$ | token-level ratio：$\rho_{i,t}(\theta)$；token-level clip | 通常保留 reference KL | 数学、代码、推理等可验证奖励任务 |
| GSPO | 在 GRPO 基础上，把 token-level ratio 改成 sequence-level ratio | prompt + 在线生成多条回答 | 需要 | 需要 Reward Model 或规则奖励函数 | 不需要 | $A_i = \frac{r_i-\text{mean}(r)}{\text{std}(r)}$ | sequence-level ratio：$s_i(\theta)$；sequence-level clip | 标准目标中不显式写 reference KL，主要靠 sequence-level clip 控制更新幅度 | 大规模推理 RL，尤其适合序列级奖励任务 |


| 方法 | 训练成本 | 工程复杂度 | 显存 / 计算开销 | 数据要求 | 适用场景 | 不太适合的场景 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| PPO | 最高 | 最高 | 需要 Policy Model、Reference Model、Reward Model、Value Model，开销最大 | 需要偏好数据训练 RM，还需要在线采样数据 | 通用 RLHF、人类偏好对齐、安全对齐、多目标优化 | 资源有限、快速迭代、小模型实验、奖励模型质量较差的场景 |
| GRPO | 中高 | 中等 | 不需要 Value Model，但需要 Reference Model、Reward Model / 规则奖励，并且要同题采样多条回答 | 需要 prompt 和奖励函数，可以是 RM，也可以是规则奖励 | 数学推理、代码生成、可验证答案任务、提升 reasoning 能力 | 开放式写作、主观评价强、奖励难设计、同组回答差异不明显的任务 |
| GSPO | 中高 | 中等偏高 | 不需要 Value Model，使用 sequence-level ratio，通常比 PPO 轻，但仍需在线采样多条回答 | 需要 prompt 和序列级奖励，适合整段回答打分 | 大规模推理 RL、数学、代码、长链路 reasoning、MoE 模型训练稳定性优化 | 需要 token 级精细控制、奖励信号非常局部、回答内部质量差异很大的任务 |
| DPO | 最低 | 最低 | 主要需要 Policy Model 和 Reference Model，不需要在线 rollout | 需要高质量 chosen / rejected 偏好对 | 偏好对齐、指令风格优化、回答质量排序、低成本对齐 | 需要在线探索、可验证奖励、多步环境交互的任务 |


## 参考

[1] Long Ouyang et al. [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155), 2022.

[2] John Schulman et al. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017.

[3] Hugging Face. [Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/zh/rlhf).

[4] Rafael Rafailov et al. [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290), 2023.

[5] Zhihong Shao et al. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300), 2024.

[6] Chujie Zheng et al. [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071), 2025.