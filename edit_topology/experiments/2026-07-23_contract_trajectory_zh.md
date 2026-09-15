# 首轮 Contract Trajectory 实验

日期：2026-07-23

案例：`remove_coca_cola_signs`

Slurm jobs：`941155`、`941165`、`941167`

## 实验问题

同一张 source image、同一个 seed 和 28 个 denoising steps 下，比较：

1. no-edit reconstruction；
2. 原始 removal instruction；
3. contract compiler 输出的 postcondition prompt；
4. 不包含 Coca-Cola 名称的 desired-state prompt；
5. 使用 target mask 后的 matched reconstruction/contract/desired-state。

保存第 `0, 5, 11, 16, 22, 27` 步的 FLUX velocity 和 predicted-clean
state。所有 masked condition 都使用 masked reconstruction 作为 trajectory
baseline；full-panel condition 使用 full-panel reconstruction。

## 关键结果

数值越低越好：

| Condition | Target 到 reference L1 | Protected 到 source L1 |
| --- | ---: | ---: |
| Reconstruction | 0.1418 | 0.0239 |
| Original instruction | 0.1695 | 0.0261 |
| Contract, full panel | 0.1487 | 0.0616 |
| Desired state, full panel | 0.1758 | 0.0568 |
| Masked reconstruction | 0.1577 | 0.0217 |
| **Masked contract** | **0.1388** | **0.0215** |
| Masked desired state | 0.1653 | 0.0219 |

最后一个 predicted-clean checkpoint 相对 matched reconstruction 的编辑能量：

| Condition | Target energy | Protected energy | Protected/target |
| --- | ---: | ---: | ---: |
| Original instruction | 0.7093 | 0.0353 | 0.050 |
| Contract, full panel | 0.0817 | 0.2494 | 3.052 |
| Desired state, full panel | 0.2610 | 0.2522 | 0.966 |
| **Masked contract** | **0.6359** | **0.0155** | **0.024** |
| Masked desired state | 0.1525 | 0.0122 | 0.080 |

## 视觉观察

- 原始 instruction 没有正确删除目标，而是在目标位置生成新的伪文字。
- full-panel contract 和 desired-state prompt 造成全局建筑/人物漂移。这不是对
  production contract path 的公平测试，因为 removal compiler 本来要求 target mask。
- masked contract 成功删除顶部大招牌，并将 protected drift 保持在 reconstruction
  水平。
- masked contract 在第二个小招牌、玻璃和人物交界处产生明显块状伪影，距离 reference
  仍有差距。
- masked desired-state prompt 没有删除顶部招牌，说明只去掉旧概念名称并不足够。

## 当前判断

第一轮支持下面两个较窄的结论：

1. Contract 必须作为 `semantic clause + spatial permission` 一起执行。只有语义 prompt
   会把修改扩散到 protected 区域。
2. Mask 能解决全局 preservation，但不能解决 target/dependent 内部的概念残留、
   边界、遮挡和纹理重建。

现在还不能声称已经发现 gradient conflict，也不能声称 ACP 有效。当前 trajectory
energy 只测量“改变了多少”，不判断改变方向是否语义正确。原始 instruction 的 target
energy 很高，但视觉上执行了错误的添加行为，这正说明下一步需要 clause-specific
semantic proxy。

## 下一步

1. 在 target 区加入 localized concept-absence proxy。
2. 在 dependent ring 加入 source/reference patch 和 edge continuity proxy。
3. 计算 semantic target gradient 与 protected/dependent gradients 的 cosine conflict。
4. 只在冲突出现的 checkpoints 运行第一个 half-space projection。
5. 使用两个额外 removal cases 和一个 addition case 检查信号是否可重复。

主结果目录：

`research_outputs/contract_trajectory_coca_cola_v3/`

## 三 seed 复现实验

补充 Slurm jobs：`941177`、`941178`

Seeds：`731301`、`731302`、`731303`

| Condition | Target 到 reference L1，mean ± sd | Protected 到 source L1，mean ± sd |
| --- | ---: | ---: |
| Masked reconstruction | 0.1562 ± 0.0020 | 0.0218 ± 0.0001 |
| **Masked contract** | **0.1414 ± 0.0273** | **0.0217 ± 0.0003** |
| Masked desired state | 0.1613 ± 0.0036 | 0.0220 ± 0.0001 |

Masked contract 在三个 seeds 上都稳定保护了 protected 区域，但 target removal
不稳定：

- `731301`：顶部招牌删除，第二个区域出现块状伪影；
- `731302`：顶部招牌删除，第二个小招牌仍有残留；
- `731303`：顶部招牌保留，第二个区域出现明显生成伪影。

因此，mask 已经解决了“修改范围”问题，但没有解决“target 应该变成什么”和“何时
停止抑制/开始重建”的问题。平均 pixel L1 改善不能替代 clause-level semantic
verification；下一阶段必须加入 localized concept-absence 和 dependent reconstruction
signals。
