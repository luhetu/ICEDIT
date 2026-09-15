# Removal 研究目标与动机收敛

检索截止：2026-09-14。本文是研究定位与实验建议，不是统一排行榜。
不同论文的输入 mask、训练数据、分辨率、后处理和评价协议不同，不能直接横向排列其分数。
“优先比较对象”不等于已证明的全球最优方法。预印本与正式发表工作分别标注。

## 结论

ICEdit 保留为工程基座和必要基线，但不能成为唯一追赶对象。
直接性能目标优先设为 ObjectClear、OmniPaint；机制近邻优先比较 Attentive Eraser、AdaEraser。
通用编辑器补充 Qwen-Image-Edit-2511 和 FLUX.2 的固定版本，检查研究问题是否仍在较新基座上成立。
不以“VLM + contract + mask + composite”作为主贡献，也不再把动态目标抑制本身当作空白。

推荐候选主题：**面向指定实例、控制附带损伤的对象删除**。
动机从“删物体时保背景”收紧为“目标与受保护内容紧邻、遮挡或共享视觉影响时，确定该删的范围并限制误伤”。
该主题仍需实测支持；mask 鲁棒性、对象影响删除和保护区域都分别有先例。

## 目标矩阵

| 对象 | 已核实定位与机制 | 对我们的作用 | 优先级 |
|---|---|---|---|
| ICEdit / In-Context Edit | NeurIPS 2025；借助上下文图像生成和 LoRA 实现指令编辑；官方另有 MoE 版本 [1] | 当前 normal LoRA 是起点；记录权重版本，不能代表整个 ICEdit 家族 | 必须 |
| 原生 FLUX.1 Fill | 当前 ICEdit 所依托的 inpainting 基座 | 隔离 diptych/LoRA 与普通局部填充的作用；不是宣称当前 SOTA | 必须 |
| ObjectClear | CVPR 2026；目标感知注意力、注意力引导融合和空间变化去噪强度；对象及影响联合删除 [2] | 最重要的直接性能与机制对手；已公开代码、权重入口和 OBER 数据 | 第一批 |
| OmniPaint | ICCV 2025；解耦 insertion/removal，处理物体及其影响，官方提供 CFD 评价 [3] | 强 removal 基线，且与 Flux 路线相关，避免只胜过弱基座 | 第一批 |
| Attentive Eraser | AAAI 2025；根据 mask 抑制前景、增强背景的自注意力重定向 [4] | 检验“关闭前景参考就够了”的简单机制 | 第一批 |
| AdaEraser | 2026-05 预印本；用参考/生成注意力相似度估计目标残留，动态调整 token 抑制 [5] | 与先前建议的动态抑制高度接近，属于必须正面对照的近邻 | 第一批，先核验可运行发布 |
| ReFocusEraser | ICLR 2026；针对小目标，结合自适应放大填充与上下文/阴影修复 [6] | 对招牌、小物件失败，检验普通放大再修复是否已经足够 | 第二批 |
| OSOR | 2026-06 预印本；一步删除、alpha head 应对不完美 mask、对象影响监督 [7] | “mask 不准”和“效率”均非空白；速度与质量的重要新目标 | 第二批 |
| Qwen-Image-Edit-2511 | 官方可下载编辑模型；较新通用编辑对照 [8] | 检验不是只修复旧 ICEdit 的特定问题 | 第二基座候选 |
| FLUX.2 | 官方提供编辑系列，模型版本/开放程度不同 [9] | 较新通用编辑水平参照；固定具体 checkpoint 或 API 版本 | 第二基座候选 |
| FreqEdit / ImageCritic | 高频保持与参考引导细节修复 [10,11] | 保护/修复模块的相关工作，不能替代专门 removal 基线 | 条件性比较 |

补充跟踪：Qwen-Image-2.0 已有技术报告，但不能由报告存在推断全部权重可本地运行 [12]。
TurboClear、FlashClear 涉及删除加速，列入后续筛查，不在未复现时引用其速度倍数为我们的结论 [13,14]。
FuLLaMa（WACV 2026）涉及 context preservation，官方论文全文在本次工具访问中失败；暂不编造细节或漏列该近邻 [15]。

## 必须修正的创新判断

1. “源图既帮助保持又妨碍删除”是有意义的研究现象，但不是新的独立发现。
2. “目标消失后减弱抑制”已与 AdaEraser 直接重叠。[5]
3. “删除阴影与反射，同时保背景”已被 ObjectClear、OmniPaint 等明确研究。[2,3]
4. “不完美 mask 的鲁棒删除”已被 OSOR 明确提出。[7]
5. “高频保护 + 空间控制 + 轨迹补偿”与 FreqEdit 直接重叠。[10]
6. “VLM 找问题再局部修复”不是 ImageCritic 之外天然成立的新贡献。[11]

## 更强的动机

删除不是把目标区域变得不像原图，而是一个具有归属关系的选择性修改：
目标实例应该消失，它造成的影响可能也应消失；邻近实例、它们的阴影、文字和结构不应随之消失。
当删除范围与受保护内容交错时，统一扩大 mask 或加强生成会增大误伤，收紧 mask 又可能留下残留。
这是本项目拟验证的冲突，而不是宣称所有现有方法都失败。

一个已有文献支持的具体难点是：ObjectClear 在复杂光照或多个阴影重叠时，可能混淆阴影属于哪个对象，
出现目标影响漏删或邻近对象影响误删。其限制部分直接讨论了这一问题。[2, Sec. 8.9]
不过，单幅图通常不能唯一确定被遮挡的真实背景，因而不能承诺“恢复真实背景”或完整物理因果识别。

可以向导师这样介绍：

> 当前对象删除方法已经能生成自然背景，但自然并不等于修改正确。我们关注目标与其他内容交错时的
> 选择性删除：模型需要移除指定实例及必要影响，同时限制对其他实例和可见细节的附带损伤。
> 我们计划研究实例归属与保护约束如何共同指导生成，而不是仅依赖单个删除 mask 或全局一致性分数。
> 首先通过同输入、同协议的对照确认该问题在强方法上仍存在，再构建相应的控制机制。

## 方法假设，而非既成贡献

将 contract 收紧为四项：目标实例、允许移除的关联影响、受保护实例/可见属性、无法可靠判断的区域。
不先追求通用 scene graph。以少量人工核验的标注建立诊断上限，随后才测试 VLM/分割器预测版本。

第一步分析 source-target 证据和相邻受保护实例证据，估计某次干预可能造成的“目标减少”与“误伤增加”。
关键问题是：这种估计是否优于单一目标残留分数，且能否处理跨实例混淆。不能只给已有 mask 预测器加新名字。

第二步只对有可靠证据支持的区域/通道进行删除或修复；不确定区域保留为显式决策问题。
完全不修改也可能是漏删，所以评价必须同时约束删除成功率和误伤，不能靠保守拒绝刷分。
最终 composite 作为工程保护单独报告；若收益仅来自贴回源图，应归为后处理收益。

## 追赶目标如何量化

- 对象删除：指定实例是否消失，是否出现替代物或重复目标。
- 影响删除：目标阴影/反射是否残留，邻近对象影响是否被误删。
- 保护：人工核验的受保护区域误差、边缘完整性、OCR 文本保持；涉及人物时另查身份/局部清晰度。
- 填充质量：边界接缝、结构连续性、纹理合理性与盲评；缺少真实背景时不把像素 L1 当唯一标准。
- 鲁棒性：对同一源 mask 做预先固定的膨胀、腐蚀、偏移；不能对每个方法挑最有利 mask。
- 资源：GPU、分辨率、精度、步数、总模型调用次数、延迟和显存一并报告。

核心比较目标：**在相近删除成功率下，减少受保护内容误伤；或者在相近误伤水平下，提高删除完整性。**
先画二者的权衡曲线，不急着把多个指标加成一个总分。对于可拒绝输出的方法，要报告覆盖率与拒绝率。

## 实验协议与优先顺序

第一阶段：完成已经提交的 15-run Coca-Cola pilot。它只判断基座、mask、prompt 和参考遮挡的影响；
遮挡输入是 OOD 诊断，不是 same-state 注意力机制证据，也不构成 SOTA 比较。

第二阶段：在已有 removal 样例和 OBER 官方评测样例的小子集上比较 ObjectClear、OmniPaint、ICEdit、
原生 Fill、Attentive Eraser。先确认模型输入所需的是对象 mask 还是对象+影响 mask，按信息预算分组。
新增受保护 mask 是额外监督，必须给可兼容基线同样的信息，或清楚区分 oracle 与预测条件。

第三阶段：只保留强基线反复失败的类型，优先检查目标紧邻人物/文字、交叠结构以及实例影响归属。
这些是现有样例上的小规模机制诊断，不是新 benchmark。使用开发子集调参数，留出未调参样例检查迁移。

第四阶段：实施一个最小干预模块，和固定抑制、固定 mask 扩张、普通 crop-inpaint、更多种子重采样比较。
同时报告官方默认配置与相近计算预算配置。ObjectClear 官方说明论文评价 guidance=1.0、demo 默认=2.5，
不能混用后声称公平复现。[2]

继续投入的条件：结果跨样例、跨种子成立；不是仅改善一种指标；保护收益在 raw 输出可见；第二基座上仍有价值。
暂停该方向的条件：强专用 removal 模型已稳定解决；简单保护 mask/局部填充就达到同样结果；
新模块依赖目标参考图、人工挑 seed 或大量额外计算才能获益。

## Sources

1. Zhang et al. In-Context Edit. https://arxiv.org/abs/2504.20690 ; official versions: https://github.com/River-Zhang/ICEdit
2. Zhao et al. Precise Object and Effect Removal with Adaptive Target-Aware Attention. CVPR 2026. https://arxiv.org/html/2505.22636 ; https://github.com/zjx0101/ObjectClear
3. Yu et al. OmniPaint. ICCV 2025. https://arxiv.org/abs/2503.08677 ; https://github.com/yeates/OmniPaint
4. Sun et al. Attentive Eraser. AAAI 2025. https://ojs.aaai.org/index.php/AAAI/article/view/34285 ; https://github.com/Alibaba-VELLDEPTH/AttentiveEraser
5. Liu. AdaEraser. May 2026 preprint. https://arxiv.org/html/2605.15921
6. ReFocusEraser. ICLR 2026. https://proceedings.iclr.cc/paper_files/paper/2026/hash/8936fa1691764912d9519e1b5673ea66-Abstract-Conference.html
7. Zhou et al. OSOR. June 2026 preprint. https://arxiv.org/abs/2606.28094 ; https://github.com/Zhouqm-Git/osor
8. Qwen official model card. https://huggingface.co/Qwen/Qwen-Image-Edit-2511
9. Black Forest Labs official documentation. https://docs.bfl.ai/flux_2/flux2_overview
10. Liao et al. FreqEdit. https://arxiv.org/abs/2512.01755 ; https://github.com/FreqEdit/FreqEdit
11. Ouyang et al. The Consistency Critic. https://arxiv.org/abs/2511.20614
12. Qwen-Image-2.0 Technical Report. https://arxiv.org/abs/2605.10730
13. TurboClear. August 2026 preprint. https://arxiv.org/abs/2608.01288
14. FlashClear. May 2026 preprint. https://arxiv.org/abs/2605.09003
15. Demir et al. FuLLaMa. WACV 2026. https://doi.org/10.1109/WACV61042.2026.00826
