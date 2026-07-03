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

## 基本概论

| 符号 | 含义 |
| :--- | :--- |
|   **state** $s$。  | 状态 |
|   **Action** $a$   | 动作 |
|   **Policy** $\pi$  | 决策函数，根据状态做出决策到控制 Agent 的动作（马里奥上下左右）|
|   **reward** $R$ | 奖励，最影响强化学习的结果 |
|   **state transition** |状态转移 $\text{old\_state} \rightarrow \text{new\_state}$,一般是随机的，随机性从环境来 |
| **Return** | 回报，把t时刻以后的奖励加起来|


### 决策函数

**Policy function** $\pi: (s, a) \in [0, 1]$,就是一个 *PDF* :

$$\pi(a \mid s) = \mathbb{P}(A = a \mid S = s).$$

强化学习就是学习这个 **Policy function** $\pi$, 来指导 Agent 观测 $\mathcal{s}$ 来做出动作 $\mathcal{a}$。这个 $\pi$ 使得Agent 动作有随机性，
这样才不会被“对手”算出来下一步动作, 所以 $\pi$ 最好是一个PDF。

{{< figure
    src="policy.png"
    caption="Fig. 1. Policy函数举例，马里奥可以上左右移动。 (Image source: [Shusen Wang YouTube, 2019](https://www.youtube.com/watch?v=vmkRMvhCW5c&list=PLvOO0btloRnsiqM72G4Uid0UWljikENlU/))"
    align="center"
    width="90%"
>}}

## Agent 与环境交互

马里奥就是 Agent，环境就是游戏界面，马里奥根据环境来做决策，然后到下一个状态。吃到金币给奖励，赢了游戏给大奖励，与之对应，输了游戏给大惩罚。

{{< figure
    src="agent-environment.png"
    caption="Fig. 2. Agent和环境交互例子。 (Image source: [Shusen Wang YouTube, 2019](https://www.youtube.com/watch?v=vmkRMvhCW5c&list=PLvOO0btloRnsiqM72G4Uid0UWljikENlU/))"
    align="center"
    width="90%"
>}}

## RL 中的随机性

* **动作随机性**：给定状态 $s$ ，Agent的状态可能是随机的，比如
    * $\pi(\text{"left"} \mid s) = 0.2$,
    * $\pi(\text{"right"} \mid s) = 0.1$,
    * $\pi(\text{"up"} \mid s) = 0.7$.

* **状态随机性**：给定状态 $S = s,A = a$ ，$S\rightarrow S'$ 环境转移是随机的，系统会随机抽样到下一个状态
    * $S' = P(\cdot \mid s,a)$


## RL 过程

* 游戏界面 (**state** $s_1$)
* 根据 $\pi$ 做出 (**action** $a_1$)
* 得到一个新的游戏界面 (**state** $s_2$) 和 **reward** $r_1$
* 根据 $\pi$ 做出 (**action** $a_2$)
* ....

{{< figure
    src="rl_tra.png"
    caption="Fig. 3. RL 轨迹链路示意。 (Image source: [Shusen Wang YouTube, 2019](https://www.youtube.com/watch?v=vmkRMvhCW5c&list=PLvOO0btloRnsiqM72G4Uid0UWljikENlU/))"
    align="center"
    width="90%"
>}}

(**state**,**action**,**reward**)轨迹：
$$s_1,a_1,r_1,s_2,a_2,r_2,\dots,s_T,a_T,r_T$$

## Return \& Reward

**Return** 回报，把t时刻以后的奖励加起来:
* $U_t = R_t+R_{t+1}+R_{t+2}+\dots$ 

*eg*, $R_t$ 和 $R_{t+1}$ 同样重要吗？
* 下面哪个你最想要？
    * 立刻给你 80
    * 一年后给你 100

很明显，肯定是现在 80 最好，一年后的 100 块不如现在的 80。所有 $R_{t+1}$ 不如当前的 $R{t}$ 重要
，一般要给 $R_{t+1}$ 打个折扣，比如八折。

**Discounted Return** 折扣回报率。
* $\gamma \in [0,1]$，折扣率 (hyper-parameter)
* $U_t=R_t+\gamma R_{t+1} + \gamma ^2 R_{t+2} + \gamma ^3 R_{t+3}+\dots$

在 $t$ 步，回报 $U_t$ 是随机的。
* 随机性来源两方面：
    * **action** 是随机的，$\mathbb{P}(A = a \mid S = s) = \pi(a \mid s)$
    * **state** 是随机的，$\mathbb{P}(S' = s' \mid S = s,A = a  ) = \pi(s' \mid s,a)$
* 对于 $i \ge t$，**reward** $R_i$ 取决于 $S_i$ 和 $A_i$。
* 因此，给定 $s_t$，回报率 $U_t$ 依赖于这两个随机变量 $A_t,A_{t+1},A_{t+2},\dots$ and $S_{t+1},S_{t+2},\dots$ 
$$A,S \rightarrow R \rightarrow U$$

## 价值函数
折扣回报 $U_t = R_t+\gamma R_{t+1} + \gamma ^2 R_{t+2} + \gamma ^3 R_{t+3}+\dots$，是一个随机变量，在 $t$ 时刻并不知道具体是什么，那如何评估呢？可以对 $U_t$ 求期望，这样就可以消除随机变量。

$Q_\pi(s,a)$，动作价值函数
$$Q_\pi(s_t,a_t) = \mathbb{E}[U_t \mid S_t = s_t,A_t = a_t]$$
动作价值函数 $Q_\pi(s,a)$ 的直观意义就是，用 **Policy** 函数 $\pi$ 在 $s_t$ 做出动作 $a_t$ 是好还是坏。  已知 **Policy** 函数 $\pi$，$Q_\pi(s,a)$ 就会给当前状态下，所有动作 $a$ 打分，这样就可以知道动作的好坏。

$Q^\star(s_t,a_t)$，最大化动作价值函数
$$Q^\star(s_t, a_t) = \max_\pi Q_\pi(s_t, a_t).$$
$Q^\star$ 直接根据 $s_t$ 对 $a_t$ 做评价，比如玩超级玛丽的时候，马里奥当前向上走“好不好”、“胜算有多大”。有了 $Q^\star$ 之后，Agent就可以根据 $Q^\star$ 对动作的评价来做决策，比如 $Q^\star$ 认为往上跳分数最高，Agent就会往上跳。

假如我们在 $Q_\pi(s_t,a_t)$ 不对 $\pi$ 求期望，而是对 $A$ 求期望呢？对 $A$ 求期望就可以把 $A$ 消除掉，是不是就可以得到环境的评估函数？

$V(s)$，状态价值函数 
$$V_\pi(s_t) = \mathbb{E}_A[Q_\pi(s_t,A)] = \sum_{a} \pi(a \mid s_t) \cdot Q_\pi(s_t, a)$$
$V_\pi(s)_t$ 的直观意义就是能够得到局势信息，根据游戏界面，我们就可以知道当前局势是不是“有利”的，是快赢了还是快输了，用于评估状态。

当然我们还可以在 $V_\pi(s)_t$ 对 $s$ 求期望得到 $\mathbb{E}_S[V\pi{S}]$，可以评估 $\pi$ 的好坏。


## How does AI control the agent?

* 假设我们有一个好的 **Policy** 函数 $\pi(a \mid s)$。
    * 观测 **State** $s_t$，
    * 随机采样得到动作：$a_t \sim \pi(a \mid s_t)$

* 假设我们知道最优动作价值函数 $Q^\star(s_t, a_t)$
    * 观测 **State** $s_t$，
    * 根据 $Q^\star(s_t, a_t)$ 选择最好的 $a_t$：$a_t = \text{argmax}_{a}Q^\star(s_t, a_t)$

所有 **RL** 的目的就是学习这两个函数，有了这两个函数之一，我们就可以让 Agent 根据 $s_t$ 来做出决策 $a_t$。