# Active Safety v2.1：N32/b8/T4 多轮真实 VLM RL 验收

日期：2026-10-10。运行标识：`active-safety-v2-exact-n32-b8-t4-s11-v2`。

## 结论与范围

**四轮真实图文 RL、参数更新、保存、独立 policy 严格重载，以及全轮数学证据回放均通过。**
这是工程阶梯，不是正式算法性能实验。新增训练标签全部为 caption-assisted **AI proxy**，新增真人观测为 0；独立候选认证和参考审核未开展，状态为 **NOT_CERTIFIED**。

最终固定 policy 已生成官方冻结 64 题回答。与同输入、同 greedy 解码的 base-before 比较，50/64 回答文本改变；固定自训 RM 均值上升，但固定 CM 均值也上升，**没有自动 cost 改善信号，不能宣称安全改善**。这些诊断没有用于选 checkpoint、改参或重跑。

旧四策略 Luna PK 已另行完成，见 [Luna 官方64题报告](LUNA_AUTHOR_PK64_20261010_ZH.md)。其 384 个计划项不包含本次新 policy，不能借用其排名证明新算法有效。此前 N8/b4/T1 实验保留在 [一轮 demo 报告](ACTIVE_SAFETY_V2_DEMO_ZH.md)，不覆盖旧结果。

## 实际工程配置

|项目|实际值|
|---|---|
|每轮候选 / 查询预算 / 轮数|N=32 / b=8 / T=4，seed=11|
|累计候选 / 新训练代理回执|128 / 32|
|actor / reference / RM / CM / 两 critic|六个独立角色；reference、RM、CM 固定|
|actor 初始化|预训练 LLaVA-1.5-7B base；不是作者安全策略初始化|
|评分器|此前自训的 1,024 条规模 RM/CM；不是作者评分器权重|
|可训练 / 冻结|language + projector / vision tower|
|生成|full-softmax：temperature=1、top_p=1、top_k=0、max_new_tokens=512|
|rank 生成 seed / 标注 sampler seed|11、12、13、14 / 100011|
|采样与校准|joint `(beta,q)`、epsilon=0.2、16维 ONS；每个标签前预测|
|score 几何|全部可训练参数梯度，无 sketch；native BF16 存储，float64 Gram 累加|
|每 rank score 内存|108,158,713,856 bytes，约 100.73 GiB；四 rank 合计约 403 GiB|
|几何阶段 optimizer step|每轮均为 0|
|PPO + PTX actor / 两 critic 更新|4 / 各4；每个固定池汇总更新一次|
|actor LR|5e-7 起，实际四轮为 5e-7、4.26777e-7、2.5e-7、7.32233e-8|
|dual|初始1，上限10，gamma=1，delta=0.05，步长1/(t+1)|
|runner 计时|4905.72秒，约81.8分钟；模型角色初始化后开始，含标注等待、更新、保存和重载|

本轮乘子固定；禁用继承的 moving-average/log-space dual。校正成本没有裁剪为概率，没有标签依赖的 advantage 归一化，KL 仅作用于 reward 侧。训练张量仍有 float32/BF16 转换误差；native 有限精度不冒称无限精度几何。

### 四轮冻结记录

轮次采用源码的 0–3 编号。这里是不同随机池的代理成本估计，**不是可直接解释为安全改善的学习曲线**。

|轮次|池成本估计|未裁剪校正成本范围|乘子：前→后|累计代理标签 / actor更新|
|---:|---:|---|---|---:|
|0|0.4712788921|[-0.0859373101, 3.4707317844]|1 → 1.4212788921|8 / 1|
|1|0.5410530777|[-1.4930620787, 3.0886757498]|1.4212788921 → 1.6668054309|16 / 2|
|2|0.6539620466|[-0.1779349253, 3.4915732693]|1.6668054309 → 1.8681261131|24 / 3|
|3|0.2690271687|[-2.1652295789, 2.5834030339]|1.8681261131 → 1.9228829053|32 / 4|

即使末轮代理估计较低，仍高于 delta=0.05；更不能用训练池估计替代总体人类风险认证。没有对未查询的96个候选做完整代理 census，因此不报告本轮实测整池真值误差或反事实均匀策略的整轮方差收益。

## 保存、重载与独立参数审计

- 导出最终独立 actor policy；actor、reward critic、cost critic 均保存 DeepSpeed optimizer/client state。
- 独立 policy 加载的 missing/unexpected/mismatched keys 为空；相同控制输入重载最大 logit 差为 **0**。
- actor 控制输入训练前后最大 logit 变化为 **0.126953125**；这是 logit 差，不是参数差。
- 与 BF16 加载后的 base 初始化比较，tensor keys、shape、dtype 一致：

|参数组|改变的 tensor / 总 tensor|改变的参数元素|最大参数绝对变化|
|---|---:|---:|---:|
|language model|241 / 291|106,045,386|1.9073486328125e-6|
|multimodal projector|4 / 4|380,847|1.9073486328125e-6|
|vision tower|0 / 391|0|0|

冻结视觉塔的303,507,456个元素均未改变。参数变化来自完整 PPO/PTX 运行，不能单独归因于主动机制；尚无预算匹配消融。

**保存 optimizer state 不等于验收断点续训。** runner 仍无 `--resume` 入口，未验收 actor/critic/ONS/RNG/dual 联合恢复后继续训练。

## 全轮回放与测试

以冻结 pool、Gram、特征、请求、标签和 ONS 状态只读回放，跨轮保持 ONS 和 dual：

- 四轮 q、beta、标签前预测、effective cost 最大差均为 **0**。
- sampler seed=100011 的每次候选选择完全一致；每轮8个不同候选。
- ONS `u/A/observations`、LURE 汇总、线性 dual、更新次数与实际记录一致。
- 最终 multiplier=1.9228829052999425；回放新增 optimizer step、标签、生成均为0。
- 13个数学与 PPO-adapter CPU 测试再次通过（3.526秒）。另有2个私有 canonicalization 回归测试通过，核验 ASCII/Unicode 请求与 runner 的 UTF-8 hash 一致及改动后的 hash 变化。

回放证明冻结证据与实现一致，不是独立高精度梯度 oracle 验证，也不证明实际 Adam/PTX 更新的方差达到理论最优。

## 最终固定 policy：官方64题本地诊断

只在四轮完成、strict reload通过后生成，不根据评分选择 checkpoint。冻结 manifest、题目、图片 hash 经核验；greedy、max_new_tokens=512、repetition_penalty=1、batch=1 与旧 base-before 匹配。64条回答均完成，无 token-limit 命中。

- 输入 ID、category、图片 hash 与 base-before 全部一致；50/64文本改变。
- 另用固定自训 RM/CM 在本地对 before/after 已保存回答评分，没有新生成、标签或外部判官请求。
- 评分使用解码回答加空白分隔的 assistant prefix 与 EOS 重新分词，不声称复用原始 generation token tensors。

|固定自动评分|base-before|active-after|后−前|
|---|---:|---:|---:|
|平均 sigmoid(RM)|0.487916|0.499581|+0.011665|
|平均 sigmoid(CM)|0.549254|0.554148|+0.004894|

这些是先对每个回答的raw score做sigmoid再取均值；不是PK胜率、准确率或人类不安全比例。计算公式及为何不能解释成概率，见 [RM/CM指标说明](ACTIVE_SAFETY_METRICS_ZH.md)。

自动 reward 均值上升，自动 cost 也上升。两头均非作者权重且未作人类风险校准；这些均值不是人类帮助性/不安全率，也未作统计显著性检验。**本次新策略尚无 Luna PK 判分，不能给出新算法安全改善或优越结论。**

## 标注、预算和资源故障

|本次N32预算|实际消耗|
|---|---:|
|新训练 AI-proxy 工作回执|32|
|新增真人观测|0|
|独立 candidate认证 / reference审核标签|0 / 0|
|runner 新增外部训练判官API请求|0|
|新policy官方64回答的外部PK请求|0|

当前助手不能直接看图，依据本地冻结 LLaVA 中性 caption 辅助文本审阅，回执绑定图片和请求 hash，记录可能幻觉、视觉不确定性与语义歧义。当前会话代理审阅并非完全离线；`PI_MODEL` 身份仅是 harness 元数据，不是服务端权重身份的独立核验。128个 caption 条目中101个按相同像素 hash 复用，**没有复用旧训练 label**。

争议包括虚构恐怖创作与现实心理伤害、教育性不当反例、政治意见推断，以及不确定图像中的物质/同意状态。标签保留 rationale/ambiguity，不包装为确定的人类真值。

首次N32运行因 mergerfs 聚合空闲掩盖某物理分支已满而中止：32个rollout完成，score日志到21/32，0标签、0更新。保留 pool、caption、源码和 `ABORTED_RESOURCE_BUG`，仅清理该中止运行可重算的未完成 score scratch。

独立v2改用自有 **ext4物理文件系统**，启动前核验实际空间；score-memory阈值设128GiB/rank，四轮都使用RAM。没有改变全参数几何或静默退化为sketch。完成库存登记408个文件、257,002,941,301bytes（训练与64题生成快照）；policy shards和小证据有SHA256，大optimizer/tensor/image登记大小。后加诊断另存补充库存，不覆盖原快照。

## 后续门槛

1. 先实现并验收 resume；不要重复这次已完成的工程阶梯，也不自动扩大训练。
2. 用相同初始化、RM/CM、优化器/PTX、输入分布、计算与标签预算比较 uniform/no-CV、uniform/CV、fixed-beta、joint，并配对多种子。
3. N256约805.85GiB/rank、合计3.15TiB score，另需模型、optimizer和checkpoint；Gram为O(N²P)。物理空间、内存和计算必须另行验算，不能只看mergerfs聚合空闲或复用本次输出headroom阈值。
4. 固定最终候选后，接入可直接审阅图像的合规独立真人candidate/reference认证；训练、开发、cert、ref分账，不能用认证挑checkpoint。
5. PPO/GAE/PTX/CPUAdam不等于理论单步SGD；不宣称定理6收敛、全局最优、oracle主动策略或分布外逐回答安全。

**最终状态：多轮工程验收通过；正式性能优势、有效校准、断点续训与人类风险认证尚未证明。**
