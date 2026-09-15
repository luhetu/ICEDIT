# ICEdit 新论文方向：导师讨论稿

## 先用一句话讲清楚

我们不先做 benchmark，也不继续把主要精力放在 parser 规则上。

我们要研究的是：

> **怎样让图像编辑模型在完成目标修改时，不破坏合同里要求保持不变的内容。**

暂定方法叫 **Active-Set Contract Projection（主动约束合同投影，ACP）**。

普通模型提出一个编辑方向。ACP 检查这个方向是否会破坏人物、建筑、文字、
纹理等保护内容。如果会，就保留有用部分，去掉会造成破坏的部分，再继续生成。

## 我们现在在哪里

已经完成：

- instruction 到 edit contract 的 parser；
- target、dependent、context、protected 四种区域角色；
- JSON validation、LLM/VLM grounding adapter、prompt compiler；
- mask、removal backend、outside-mask exact composite 和 demo；
- addition/removal 的旧失败案例；
- 当前本地测试 87/87 通过。

但是这些只能说明系统结构是稳定的，不能说明我们已经有论文贡献。现在缺少的是：

> contract 是否能够直接改变模型的 denoising 过程，并解决 raw output 的失败。

## 为什么不继续把 router 当主创新

最近已有工作在做：

- 按任务类别选择 attention 操作；
- 在 Text、Mask、Reference、Base experts 之间动态 routing；
- 用 causal information-flow mask 控制 source/reference；
- 用 VLM 做 planning、verification 和多轮 correction；
- 用 reference alignment 修复 texture/detail。

所以 `contract + router`、`VLM verifier` 或 `texture keeper` 单独拿出来，都容易被
认为是已有模块的组合。它们可以使用，但不能是论文最中心的贡献。

## 我们的新核心假设

一个编辑同时有两种要求：

```text
目标要求：必须改变什么
保护要求：什么不能变坏
```

例如：

```text
删除 Coca-Cola signs
保持建筑结构
保持人物
保持其他文字
允许 signs 后面的墙面被重新生成
```

普通 editor 只有一个整体 guidance。这个 guidance 可能一边帮助删除 sign，一边又
让 sign 重新出现，或者破坏建筑和其他文字。

我们的假设是：这种冲突可以在 denoising 过程中被测量。

设正常编辑更新为 `d_edit`，第 `i` 个保护条款的 violation loss 为 `L_i`：

```text
conflict_i = <grad L_i, d_edit>
```

如果 `conflict_i > 0`，表示当前编辑方向会让这个保护条款变得更差。

## 方法怎么做

### 1. Contract 变成可计算条款

Parser 输出的不再只进入 prompt。每个条款还会得到：

- 条款类型：change 或 preserve；
- 对应区域：target、dependent、context、protected；
- 可计算 proxy：DINO、CLIP/SigLIP、edge/depth、OCR 或 detector；
- hard/soft constraint；
- 当前是否 active。

### 2. 只激活正在出问题的约束

一个保护条款满足下面任意条件时才 active：

- 已经被破坏；
- 下一步预计会让它变坏；
- 第一轮输出被 verifier 判定失败。

这样不会因为全局 preservation 太强而阻止模型完成编辑。

### 3. 对编辑方向做投影

我们求一个离正常编辑方向最近的新方向：

```text
minimize    ||d - d_edit||^2

subject to  目标条款继续变好
            active 的保护条款不能变坏
            更新不能超过 trust region
```

有真实冲突时加入 slack，避免问题无解。第一版只在 3-5 个 denoising checkpoints
运行，并在少量 candidate fields 上求系数，不做昂贵的全像素大规模优化。

### 4. VLM 做证据，不直接负责生成

第一轮生成后，VLM 输出：

```text
pass / fail / uncertain
失败条款
失败类型
confidence
evidence box 或 mask
```

第二轮只重新激活失败条款，并继续保护已经通过的 hard invariants。OCR、detector、
pixel 和 feature tools 能判断的内容，不全部交给 VLM。

### 5. Texture 放在正确的位置

Consistency Critic 的 reference detail alignment 可以成为一个 specialist constraint：

- protected 区域保持 source texture；
- reference texture 只进入合同允许的区域；
- removal/replacement 时旧 target texture 必须消失；
- 暴露出来的表面使用 context texture 重建。

Texture 不是主创新，合同约束投影才是主创新。

## 第一张最吸引人的图

使用 Coca-Cola failure：

1. source image 和四种 role masks；
2. ICEdit raw failure；
3. denoising 中 target/protected conflict heat map；
4. ACP 去掉的 harmful update component；
5. ACP raw output 和每个 clause 的 pass/fail。

再用同一个 sign 做四种操作：

```text
add | remove | replace | preserve sign but change wall color
```

这张图要说明：同一个视觉概念在不同合同里有不同权限，方法不是看见某个 noun 就
固定增强或抑制它。

## 不做 benchmark，先做什么

先固定 24 个 diagnostic cases：

- 8 个 removal；
- 6 个 addition/attachment；
- 6 个 replacement/attribute；
- 4 个 text/texture preservation。

这些不是新 benchmark，只用于快速判断核心假设是否成立。

必须比较：base ICEdit、contract prompt、mask/composite、guidance search、static
routing、PCGrad，以及 ACP。必须看 raw output，不能只看 composite 后的结果。

## 继续还是停止

只有满足下面条件才扩大研究：

1. conflict score 能预测真实失败；
2. ACP 同时改善 target success 和 preservation 的 Pareto frontier；
3. 至少三个 edit topologies 有提升；
4. raw output 确实改善；
5. 打乱 clause、mask 或 projection direction 后提升消失。

如果 mask/composite 已经解释全部提升，或者 static rule 和 ACP 一样好，就停止这个
claim，不浪费时间做大实验。

## 四周安排

第一周：12 个 case，记录 denoising checkpoints，画 conflict trajectory。

第二周：实现最小 ACP，只用一个 target proxy 和一个 invariant proxy。

第三周：加入 OCR/texture clause、VLM evidence 和一次 correction。

第四周：扩展到 24 个 case，做 shuffled controls、Pareto plot 和 tutor/paper figures。

四周以后，只有核心方法通过上述检查，才跑现有公开 benchmarks。暂时不创建新的
benchmark。

## 对导师可以这样说

> Existing image editors treat an instruction as a soft preference. We treat it
> as an executable contract. Our method projects each risky denoising update to
> the nearest direction that still makes the requested edit while preventing
> localized invariant regression.

这项工作达到 CVPR 水平的关键，不是系统模块多，而是要证明：

> contract conflict 是真实、可重复的生成机制问题，而且 ACP 比 prompt、mask、
> global guidance 和 routing 更有效。

如果想投 NeurIPS，还需要进一步给出 active-set feasibility、stability，以及基于 smooth
proxy loss 和 trust region 的 first-order no-regression 分析。
