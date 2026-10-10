# Active Safety v2.1：真实 VLM RL demo 验收

日期：2026-10-10。该实验独立于旧复现、四策略回答和官方 PK，不覆盖旧结果。

本文保留 N8/b4/T1 的历史结果。后续 **N32/b8/T4 多轮工程阶梯已完成**，见 [独立多轮报告](ACTIVE_SAFETY_V2_N32_ZH.md)；两个实验的预算与诊断分账。

## 结论

**真实图文强化学习闭环已完成一轮，并通过独立严格重载。**
完成的是工程与校正信号验收，不是正式算法性能实验，更不是人类风险认证。
固定样例的自动评分没有改善，因此不能把这轮 demo 写成安全性或帮助性提升。

## 新增实现

- `safe_rlhf_v/utils/active_safety.py`：受下界约束的 water-filling、中心化精确 Gram 几何、联合 `(beta,q)` 优化、LURE、式 (9) 校正成本、ONS、投影线性对偶、投注风险上界、参考策略精确二项上界、认证混合权重。
- `safe_rlhf_v/trainers/text_image_to_text/active_safe_rlhf_v.py`：PPO adapter；当前池固定乘子，不做标签依赖的 advantage 归一化，不裁剪校正成本，KL 仅属于 reward 侧。
- `safe_rlhf_v/configs/train/active_safety_v2.yaml`：推荐参数清单；**不是旧 trainer CLI 可直接加载的完整原生训练配置**。操作 runner 保留在私有研究目录，复用已跑通的数据、六角色模型与 DeepSpeed 工程。
- `tests/test_active_safety.py`、`tests/test_active_safety_ppo.py`：数学和 adapter CPU 测试。

adapter 本身不负责生成池、排队标注或禁用原生对偶更新；这些由独立 runner 编排。不能把 adapter 单独替换进旧训练入口并宣称已实现完整主动算法。

## 本次真实训练

运行标识：`active-safety-v2-exact-demo8-b4-t1-s11-v3`。

| 项目 | 实际结果 |
|---|---|
| 候选 / 新训练标签 / 更新轮数 | N=8 / b=4 / T=1 |
| actor / reference / RM / CM / 两 critic | 六个独立角色；RM、CM 和 reference 固定 |
| trainable / frozen | language + projector / vision tower |
| 生成策略 | temperature=1、top_p=1、top_k=0、max_new_tokens=512 |
| 每 rank 策略 RNG seed | 11、12、13、14 |
| score 几何 | 全可训练参数梯度，不使用 sketch；native BF16 分区，float64 Gram 累加 |
| 几何阶段 optimizer step | 0 |
| 每 rank score 存储 | 27,039,678,464 bytes，约 25.18 GiB |
| 汇总 actor PPO+PTX 更新 | 1 |
| 两 critic 更新 | 各 1 |
| 校正成本均值 | 0.5302411031 |
| 校正成本范围 | [-0.6530351425, 1.5845062318]，没有裁剪为概率 |
| 对偶乘子 | 1 → 1.4802411031，delta=0.05 |
| 独立严格 actor 重载 | 通过；同控制输入最大 logit 差=0 |
| actor 控制输入前后 logit 最大变化 | 0.125；这是 logit 差，不是参数差 |
| block wall time | 748.84 秒；含交互标注等待、保存和重载，不是纯计算吞吐 |

保存了 actor 及两 critic 的 DeepSpeed checkpoint（含优化器、轮数、乘子、ONS），并导出可独立加载的 actor policy。**尚未验收断点恢复后继续训练**，不能将“保存 optimizer 状态”写成“已验证 resume”。

### 独立参数审计

比较导出 BF16 权重与 **BF16 加载后的** base 初始化，而不是将原始磁盘 dtype 转换差异误算成训练更新：

| 参数组 | 改变的元素 | 最大参数绝对变化 |
|---|---:|---:|
| language model | 66,934,838 | 9.536743e-7 |
| multimodal projector | 270,582 | 9.536743e-7 |
| vision tower | 0 | 0 |

确认 actor 的实际参数发生更新，并且冻结视觉塔没有变化。

## 校正与方差诊断

训练结束后另审阅剩余 4 个候选，仅用于完整有限池诊断，不加入训练，也不用于选择 checkpoint。

- 8 个池候选的 AI-proxy unsafe 均值：0.5。
- 主动取得的 4 个标签的简单均值：0.75；这不是主动采样下正确的池风险估计。
- LURE 校正估计：0.530241；这一次有限池绝对误差为 0.030241。单次误差小不构成无偏性或精度的实证证明；无偏性由枚举数学测试核验。
- 在本轮初始池，使用完整 AI-proxy 标签和实际 Gram 计算的首查询条件方差，相对均匀选样、beta=0 的比值为 **0.3703**，即该固定池、该代理标签下低约 63%。
- 随后三个已观测自适应历史上的条件方差比值为 0.4008、0.3184、0.3762。它们**不是**反事实均匀策略的整轮方差，也不是跨运行平均收益。

ONS 仅见 4 个标签，预测仍非常接近 0.5。此 demo 不证明校准学习已经有效，也不证明达到 oracle。

## 固定训练样例的前后生成

对同一组 8 个图文输入，以相同 greedy 解码、max_new_tokens=512 做 before/after 生成，固定 RM/CM 分别报告，不合成加权榜。输入来自训练源，共 7 个不同源行；重复输入是独立池抽样产生的，不是 8 个独立验证题。

| 指标 | before | after |
|---|---:|---:|
| 平均 sigmoid(RM) | 0.514560 | 0.476905 |
| 平均 sigmoid(CM) | 0.724637 | 0.735836 |

5/8 回答文本发生变化。**reward 下降、自动 cost 略升，没有观察到自动评分改善。**
这是小训练样例上的确定性生成诊断，不是 held-out 评测；CM 也不是经过认证的人类风险。
没有根据这些结果挑选 checkpoint、追调学习率或重跑以制造正向结果。

## 标注来源、预算与限制

用户要求当前助手做代理标注，但图片读取接口返回“当前模型不支持图像”。因此本次使用：

1. 私有机器上的冻结 LLaVA 生成中性图像描述，绑定图像 hash；
2. 当前交互助手审阅问题、回答与描述，逐条给出二元标签及理由；
3. 回执明确记录 `ai_proxy`、无直接视觉输入、局部描述可能幻觉，以及歧义。

不能声称助手直接看过图像，不能将这些标签冒称真实人类观测。`PI_MODEL` 仅是 **harness 元数据标识**，不是对外部服务实际权重身份的独立核验；私有回执中的身份未核验说明保留，不公开具体会话元数据。
本地 caption 只是视觉辅助，不是替代当前助手的安全判官。没有另行向外部判官 API 发送训练数据；当前会话的代理审阅是用户要求的临时代标流程，不能把“runner 未调用外部判官”扩写成“当前助手运行完全离线”。

| 预算项 | 实际消耗 |
|---|---:|
| 完成运行的训练 AI-proxy 回执 | 4 |
| 已中止预检的 AI-proxy 回执 | 2 |
| 训练后完整池审阅新增 AI-proxy 回执 | 4 |
| 总 AI-proxy 标注回执 | 10 |
| 真实人工观测 | 0 |
| 独立候选认证 / 参考认证标签 | 0 / 0 |
| runner 新增外部训练判官 API 请求 | 0 |

预检与最终运行存在相同回答的重新审阅；不将重复内容当作额外独立人类信息。计数是实际标注工作回执，不是 10 条独立人类样本。
认证预算 1024/512 仅为计划，**NOT_CERTIFIED**；未计算或宣称真实人类风险上界，也未部署未经独立认证的混合策略。

## 预检 bug 与记录保留

- v1：`GenerationConfig` 隐含 top_k=50，与所求 full-softmax log-prob score 不一致。未标注、未更新；显式修复 top_k=0 后开新目录。
- v2：相同 rank seed 导致各 rank 的生成流相关。已取得 2 个代理回执，但未更新；保留回执，修复 seed+rank 后开新目录。
- ZeRO-3 精确 score：显式清空梯度分区缓冲，抵消 accumulation 的 loss scaling，几何阶段不 step；复制同候选到所有 rank 后按参数分区计算 Gram 并 all-reduce。
- 数学输入边界：拒绝 NaN、非法概率、非整数预算、不完整池分区及会导致无限搜索的 tolerance=0。

两版预检被明确标记为中止，不覆盖、不改为成功、不自动重试。旧 Qwen 两个判分任务按用户要求停止，部分回执保留；旧训练/回答/PK 数据不受这轮 demo 影响。

## 测试与复核

在已安装项目依赖的环境运行：

```bash
python -m unittest discover -s tests -p 'test_active_safety*.py' -v
```

**13 个测试通过**：自适应多查询无偏性与鞅交叉项、精确方差、LURE 退化/census、Gram 几何、联合优化与 SciPy 数值对照、完美 proxy、ONS 投影/预测时序/遗憾检查、投注范围与 mixture、参考二项上界与 SciPy 对照、非法输入、未裁剪成本与固定乘子 adapter。

最终数学模块增加了输入防护与参考上界函数。以完成运行的冻结 Gram、特征、请求和标签重新回放，q、beta、校正成本与实际运行逐项一致，最大差均为 0；该回放没有新标签或优化器更新。

## 扩大实验前的门槛

1. 后续 N=32、b=8、T=4 工程阶梯已通过，见上述独立报告；这不是理论默认设置，也不是根据评分调参。resume 仍须实现并验收，再进入理论建议的 N=256、b=16、T=32。
2. exact score 存储按候选数线性增长：N=256 约 **805.85 GiB/rank、合计 3.15 TiB**，另需模型、优化器、候选、checkpoint 空间；Gram 计算还随 N² 增长。runner 有 native-dtype disk memmap fallback，不允许静默替换为 sketch。
3. 正式实验须接入支持直接图像审阅的合规标注流程；真实人工认证必须使用目标人类标注，不能沿用本次 caption-assisted AI 标签。
4. 保持同初始模型、reward/CM 映射、计算量与标签预算，比较 uniform/no-CV、uniform/CV、fixed-beta active、完整 joint；另设零训练标签的固定自动成本参考，并配对种子。
5. 独立认证池与参考审核在最终候选选定后执行；所有开发、训练、认证、参考标签分账，不能用认证结果反向挑 checkpoint。
6. PPO/GAE/PTX 不是式 (22) 的单步 SGD，不引用定理 6 作为此次 PPO 优化的收敛保证。native 有限精度、经验池 score bound 也不等同于已证明全局神经策略常数。

**本文验收状态：一轮真实 RL、参数更新、保存及严格重载通过；后续四轮工程结果另文报告。正式性能、校准收益、多轮 resume 和人类风险认证仍待实验。**
