# Joint-Gradient 接续交接：项目进展、用户偏好与执行边界

> **后续状态已更新：** 请先读 [22:55真实一步完成后的交接](CODEX_HANDOFF_20261016_POST_UPDATE_ZH.md)。现已提交从原基线训练一步的checkpoint；包含保存失败尝试累计发生2次actor step，本轮4次真实反馈成功。用户已放宽标注范围并要求自主继续。下文保留21:06历史快照，不再代表当前运行或授权状态。

> 状态核对时间：**2026-10-16 21:06:53（UTC+8）**。这是运行中的时间截面，不是任务完成声明。当前真实策略更新 **0**；本轮新反馈尝试 **0**；此前独立 SOL campaign 已完成 **196 次成功标注**。
>
> **公开脱敏交接，仅发布文档。** 新实现、测试、原始图文、响应、账本、运行日志和权重均留在私有工作区。本文的私有文件名只是定位键，不代表 GitHub 中已有这些文件。不得根据本文猜测服务器地址、复制凭据或公开私有包。取得获准私有工作区后才能执行。
>
> 本文取代 [10月13日交接](CODEX_HANDOFF_20261013_ZH.md) 中的“当前状态/下一步”，不改写历史实验事实。负责人摘要见 [10月16日进展](../PROGRESS_20261016_ZH.md)。

## 1. 接手先记住这五件事

1. **首要目标是完成一次用户自己的 Joint-Gradient 真实更新。** 不是再做一遍未修改的作者方法，也不是无限扩展审计。最终路线仍为真实 A → 留置 B → 生产验收 → 正式 C/D；本次有限探索不等于这些阶段已通过。
2. **持续自主工作。** 用户明确要求“干一段可以给个报告但是要自己一直干啊”。报告是进度通知，不是停止点。不要在每个小阶段完成后等用户再说“继续”。
3. **SOL 授权不是当前瓶颈。** 用户已授权在当前实验范围内按需使用 `gpt-6.1-sol`，不要反复询问已授权调用或以费用不确定为由停工。授权仍受 **32 条现有新回答 / 最多4次新反馈尝试 / 最多1次真实 actor 更新 / 0自动外部重试** 约束。
4. **当前有运行中的自有任务，不是空闲资源。** v3 整池实际梯度计算最后观察到 **1040/1184 次反传**；结束后有一个仅依赖该自有任务的容量检查脚本。接手先查状态，不要重复启动、覆盖输出或抢占其锁。
5. **尚未训练成功。** 梯度、校准器更新、CPU测试、forward、文件hash和启动成功，都不算策略权重更新。只有实际 optimizer step 的证据才能改变更新计数；checkpoint 提交另行核实。

## 2. 用户偏好与协作方式（必须传递给下一会话）

| 偏好 | 接手者应如何执行 |
| --- | --- |
| 持续推进，不要汇报后停工 | 已授权范围内自己规划并执行；运行等待期间做独立准备。仅在实质新授权、安全资源不可用、不可恢复错误或冻结方法冲突时停下说明。 |
| 先完成一次真实更新 | 优先处理实际梯度、真实反馈、更新容量与一步训练；论文细节、额外审计、反复封存不能取代主线。 |
| 做自己的方法，不是仅复现作者流程 | 保持一般八状态选择、真实校正、合并优势、同参数点 PPO+PTX、一次 actor step、式(15)乘子更新。 |
| 反馈资源已授权 | 在4次尝试上限内自行使用 SOL；不重复请求相同许可，也不把宽松费用要求理解成无限调用。 |
| 用平实中文说清真实状态 | 报告先写实际权重更新数、新反馈尝试数，再写当前障碍和下一具体行动。不要用测试数量掩盖训练未发生。 |
| 复用已完成资产 | 不重新生成 fresh32、不重复历史预热、不重训已有 scorer，不用旧回答的标签给新回答打标签。 |
| 保持科学与失败记录诚实 | 失败版本、部分输出、异常和未过门禁均保留；不事后换阈值、挑赢家、倒置符号或把探索写成正式收益。 |
| 尊重资源及隐私 | fresh资源检查、协作锁、既有环境；不共享占用GPU、不查看或终止他人任务、不自动安装依赖或搬凭据。 |
| GitHub同步方式 | 用户本次选择：现有仓库和分支直接提交、推送脱敏文档；不发布服务器信息、凭据、私有实验代码/数据/权重。 |

持久私有偏好文件：`OPERATOR_CONTINUOUS_WORK_V1.json`。最近接续摘要：`LIVE_TRIAL_CONTINUATION_V2.json`，但其中运行计数是早期快照；**实时日志优先于它的“最后观察到”字段**。

## 3. 当前状态总表

| 工作 | 已完成的真实内容 | 不能宣称的内容 |
| --- | --- | --- |
| 既有 SOL 标注 | 4 pilot + 192 main，196次成功；无自动重试。真实图片、问题和当时回答绑定，startup64/dev128隔离 | 不等于剩余机制256已标注；不是本轮fresh32标签 |
| 四 scorer 质量 | startup64拟合固定映射，dev128只评估；两档CM通过暂定数值筛查，两档RM未过附加绝对帮助性筛查 | 不选共同CM赢家，不证明标量奖励不能支持论文原RL |
| FP32真实7B精度 | 相同BF16初始权重拓宽到FP32，多种真实backward/流式路径及重复检查完成 | BF16历史失败没有被“追认通过”；生产所有门禁未自动通过 |
| fresh32 | 16危险+16良性，32条新回答、固定low RM/CM评分、token/图片绑定完成 | 无重新生成/择样；它们还没有本轮真实SOL反馈 |
| 校准器初始化 | 历史128 → startup64，一次通过共192个观测；固定32维特征与映射完成 | 不是策略训练，不是历史协议和SOL协议等价 |
| 策略/critic基线 | 旧概率BF16逐token回放；当前FP32概率和两个critic值已完成 | 旧概率是事后回放，不是保存的原生成logits |
| 全参数梯度 | 6,759,919,616个可训练参数；最长fresh样本八状态与重复验证完成；整池v3运行中 | 1040/1184不是完整Gram，不能提前按不完整几何选反馈 |
| 本轮反馈 | 32个真实图片+新回答请求体准备；独立4次账本和顺序查询runner完成，mock联调通过 | 实际新尝试0；没有真实4标签或196观测在线更新结果 |
| 一池更新程序 | actor/PPO+真实PTX、双critic、dual、CPU AdamW、checkpoint读回程序已编写并上传；聚焦CPU测试通过 | 真实容量检查未完成，实际actor/critic更新均未完成 |
| 正式验收 | 仍保留门禁与失败 | 无正式A/B、C/D、安全性提升或反馈效率优势认证 |

不同版本的测试套件有重叠，不能将其数量直接累加为独立测试总数。

## 4. 已取得的关键结果与解释边界

### 4.1 scorer与真实标注

- 既有196次调用是已经完成的历史事实。旧文档“SOL调用0/发送器未联调/等待首次标注”不再是当前状态。
- dev128上，两档RM固定映射的 Spearman 分别约 **−0.1866 / −0.0741**；MSE **0.08365 / 0.08608**，常数基线 **0.08203**。两者未通过附加绝对帮助性校准筛查；不重拟合、不翻转符号、不事后选有利子集。
- 两档CM的 AUC **0.9229 / 0.9392**，balanced accuracy **0.9212 / 0.9017**，Brier **0.07158 / 0.07493**，常数基线 **0.15869**。均通过暂定数值筛查，不构成共同CM选择或独立正式验证。
- 相同官方590对的CM检查已完成：low/high排序准确率 **442/590 / 470/590**，安全符号准确率约 **0.8449 / 0.8619**。它是历史集描述性比较，不是盲测胜者证据。
- RM训练目标经实际源码核查是pairwise logistic加score正则，不是0–3绝对评分回归。**成对排序与跨问题绝对评分是不同目标**；上面的失败不能直接推出原论文RL无效。

### 4.2 数值配方与本轮输入

用户已选择并冻结 `TRIAL_UPDATE_RECIPE_V1.json`：

- `fp32_verified`：从固定BF16权重拓宽，实际backward/训练用FP32；不以FP32全序列当前概率替换旧BF16概率。
- `legacy_constants_own_objective`：复用学习率 `5e-7`、KL `.02`、GAE `.95`、PTX权重 `16`、critic初始化等常量，**不复用作者的分开actor/PTX更新目标**。
- AdamW：betas `(.9,.95)`、eps `1e-8`、weight decay `0`、梯度裁剪 `1`；最多一次actor及各一次critic更新。
- 新32条上的 current/old 比率范围 **0.742881–1.311655**，有 **3个自动gate/clip token**。不能假设ratio全部为1或gate全不活跃。
- 长度最长fresh样本上下文872、回答256。v3容量检查11次真实backward完成，重复梯度逐元素相同；十个梯度范数与既有版本相同，actor不变。**这不代替最长PTX及全优化器容量检查。**

### 4.3 数据与PTX

- PTX全13分片、103,012行已筛查；保守排除 **934行 = 922精确重叠 + 12超大未知项**。这不是“family完全干净”的正式split接受。
- PTX绑定读取与实际tensor消费已有CPU验证；本轮配方使用按分片/行字典序选出的前32个有效且未排除样本，无截断；无效跳过须在optimizer前记录。
- 特征来自固定原始base的最后实际回答token的final-normalized hidden state，而非变化的actor或scorer特征。探索性32维表示已固定，不改映射。
- 人工视觉审阅仍为 **0**；原始family、完整权利链、最终split、独立方向及final test仍未接受。OCR、contact sheet、provider理由或模型处理图片都不等同人工视觉接受。

## 5. 整池梯度运行：版本、失败与当前后台依赖

| 版本 | 状态 | 保留的教训 |
| --- | --- | --- |
| v1 | 巨大缓存导致严重减速，完成6/608次反传后，仅终止核实属于自己的进程；无完整Gram | 容量单样本耗时不能预测巨大缓存行为 |
| v2 | 私有非THP缓存修复后复制可持续，但点积/串行norm过慢；再次显式停止自有进程，部分证据保留 | THP/NUMA是可能因素，不是已证明根因 |
| v3 | 私有缓存16个H向量，32个worker处理不相交坐标chunk；**1040/1184**，完整DONE尚未观察到 | 单个einsum与chunk加法次序不变；不做sketch、仿射捷径或参数子采样 |

最近一次记录：sample13、G state7，单次计算约3.45秒，覆盖全部6,759,919,616参数，当前logprob精确相同；已观察到 `BLOCK-16.json`（16/64个H列、320次反传时的块完成标记）。这些是部分进度，不是完整最终验证。

**后台依赖已登记：** `after_own_gram_capacity_v1.sh` 等待当前自有v3梯度launcher退出；仅在完整Gram成功、1184计数及零更新等条件成立后，启动 `real_one_pool_update_v1.py capacity`。容量程序重新做UUID/空闲资源/锁检查；遇占用立即失败，**不是等待他人GPU或自动抢卡**。容量模式不执行任何optimizer step。

接手时先检查：

- `REAL32_GRADIENT_V3_LAUNCH_PID`、`real32-gradient-full-v3/BACKWARDS.jsonl`；
- `REAL32_GRADIENT_FULL_V3.exit`、`GRAM.json`、`CONTRACT.json`、`DONE.json`、`FAILED.json`、`RESOURCE_ABORT.json`；
- `GRAM_THEN_CAPACITY_V1.pid/.log/.exit` 和 `REAL_ONE_POOL_CAPACITY_V1.exit`；
- `real-one-pool-capacity-v1/{PTX_SELECTION,CONTRACT,DONE,FAILED}.json`。

**不存在exit/DONE不等于成功，也不必然等于进程还活着。** 需核实自有进程与日志；不检查他人argv/env，不启动重复job。

## 6. 接下来按这个顺序执行，不要另开审计主线

### 第一步：确认完整实际梯度和真实容量

核对1184次调用、所有八状态/完整参数坐标、固定actor点、原回答token/gate、重复检查、source manifest和最终Gram/DONE。回传具名产物并校验hash。保留被停止的v1/v2，不重写为成功。

容量模式检查最长选中PTX、两个critic的最长fresh backward，以及三套全参数CPU AdamW状态的实际内存。当前只知道程序已写并有聚焦CPU测试，**没有真实容量成功结果**。失败时先留证再做版本化工程修复；不擅自降维、改目标、改精度或超资源上限。

### 第二步：执行最多4次顺序真实反馈

复用 `fresh-feedback-v2/` 的32个body和独立授权；原196次账本不修改、不重置。`run_trial_feedback_v1.py` 用完整实际Gram，在每次查询前保存：剩余集合、8状态概率、q/β/ω、历史reveals、ONS、RNG、body/Gram/特征绑定及label-unknown证据。

然后持久预留尝试，再调用现有一次性SDK。实际回执后才更新ONS，权重为 `1/(M*q_selected)`；移除样本，重算下一次计划。4次成功后保存 `ACTUAL_REVEALS.json` 与 `ALIGNED196_STATE.json`。

- 既有ledger测试和完整4步mock流程已通过；mock不是实际标签。
- 失败、超时或pending消耗预算并使本campaign不能隐式继续；无自动重试/重发/重置。
- API凭据留在获准发送环境，不随数据传到GPU主机。
- 编码图片文件SHA与canonical RGB SHA不同，必须各自核对，不能交叉比较。

### 第三步：一次真实Joint-Gradient更新与checkpoint提交

`real_one_pool_update_v1.py update` / `run_real_one_pool_update_v1.sh` 已实现、尚未实跑：

1. 固定fresh32 PPO和32个真实PTX样本；校正后的PPO与causal-CE PTX在同一actor参数点累计。
2. 完整梯度覆盖、裁剪、CPU offload不改变参数点；actor执行恰好一次AdamW step。
3. 双critic使用固定proxy returns，各执行一次step；不用revealed标签直接替换critic目标。
4. 使用按实际揭示顺序构造的、未裁剪校正风险更新式(15)乘子，不信任事件中缓存的risk值。
5. 保存权重、optimizer、scheduler、RNG、dual、cursor和绑定；逐张量读回检查，再提交checkpoint目录。

CPU已检查：实际32条padding/objective一致性、预分配AdamW与stock一步结果相同、JSON历史键恢复、进程局部THP设置、tiny模型和优化器精确读回。**这些不认证真实7B offload、峰值容量或strict resume。** 即使读回成功，也不能宣称分布式strict resume已通过。

出现step中途失败时，更新次数可能不确定；不得删除保留标记后重跑。区分“发生过step”“完整pool结束”“checkpoint已提交”三件事。

### 第四步：汇报并继续

用中文报告真实策略更新数、新反馈尝试数、checkpoint状态、失败或未过门禁。完成探索一步后再按实际证据安排A/B及生产门禁工作；不能自动扩大到正式C/D、剩余256标签或多步训练。

## 7. 冻结方法合同（不可为了尽快运行而改变）

- `GENERAL_EIGHT_STATE_EQ6`：R∈{0,1/3,2/3,1}，C∈{0,1}，状态顺序 `[(r/3,c) for r in range(4) for c in range(2)]`；全8状态、全可训练坐标、同actor点、固定token和gate。
- PPO先合并优势 `(A_R−νA_C)/(1+ν)`，再clip到 `[.8,1.2]`；response均值、reward-only KL、不做advantage批归一化。
- Eq6基于真实梯度Gram，剩余集合居中的自动梯度，β∈[0,1]²、概率floor ε/M，ε=.2。不能以scalar score或三个仿射基底替换一般八状态。
- 历史128为原低数据顺序前64个共享pair的双方回答；显式helpfulness和yes/no安全标签，不从preference ID推断。之后startup64对齐；dev128只评估。
- JointONS：32维、8头、ε=.2、ξ=.1、H0=25I、w0=0；raw线性head平方损失；历史/启动unit weight，在线用1/(Mq)。
- 式(15)：ν0=1，范围[0,10]，δ=.1，α=1/(completed_updates+1)，未裁剪实际校正风险。
- 不把旧作者分开的RL/PTX step命名为Joint-Gradient；不复用旧回答标签、重生成结果或选择性丢弃长回答。

## 8. 证据索引与历史状态纠正

以下为私有定位键，**不要将对应私有文件直接加入Git提交**。

| 定位键 | 用途 |
| --- | --- |
| `LATEST_CONTINUATION_20261016_V9.json` | 最新完整历史封存；新试验工作为追加证据，不是新的完整封存 |
| `OPERATOR_SCOPE.json` / `OPERATOR_CONTINUOUS_WORK_V1.json` | 当前实验范围与持续工作要求 |
| `TRIAL_UPDATE_RECIPE_V1.json` | 已批准FP32/常量复用/自身目标的一步探索配方 |
| `SOL_REAL_192_AND_RM_SIGNAL_20261016_ZH_PRIVATE.md` | 历史196标注与192主样本质量结果 |
| `calibrator-initialized-v1/` | 固定特征、历史128、对齐192及初始fresh预测 |
| `REAL32_FROZEN_PPO_V1.json` / `policy-*-v1/` | fresh32真实概率/critic/gate与冻结pool |
| `real32-gradient-capacity-v1/` / `real32-gradient-capacity-v3/` | 实际最长fresh容量与精确重复证据 |
| `real32-gradient-full-v{1,2,3}/` | 两次停止的部分证据与当前v3整池运行 |
| `fresh-feedback-v2/` | 32个请求体、4次授权；本次快照尚无实际campaign账本 |
| `trial_feedback_ledger_v1.py` / `run_trial_feedback_v1.py` | 独立尝试账本、真实顺序选择与反馈流程 |
| `one_pool_torch_engine_v2.py` / `real_one_pool_update_v1.py` | 更新引擎、真实模型/CPU优化器/PTX/checkpoint接入 |
| `REAL_ONE_POOL_TESTS_V1_WITH_READBACK.log/.exit` | 5项聚焦CPU控制，通过不代表真实更新发生 |
| `TRIAL_FEEDBACK_FLOW_TEST_V1.log/.exit` | 合成几何+mock回执的4步联调，实际API调用0 |

必须纠正的旧状态：

- 旧“新增SOL0”仅描述当时或本轮fresh试验，不能抹掉已有196次成功。
- 旧“反馈未授权/费用材料不足所以不能推进”不应继续阻挡当前32/4/1/0探索。
- 旧“特征/历史预热/概率基线未完成”已过时，不要重做。
- 旧“v2在跑/资源已释放”已过时；本次实时核对是v3在跑，且有自有任务容量依赖。
- 旧BF16差异、数据接受缺口、正式训练门禁仍有效，不能因为探索获准而消失。

## 9. 仍未解决的正式门禁

真实新A、留置B、生产一池更新验收、严格checkpoint恢复、正式共同训练配方、训练/PTX split接受、独立评估方向、人工视觉和权利链、最终测试及C/D训练长度/种子/反馈范围仍未完成。共同CM未选赢家；额外RM校准筛查失败仍需如实保留。

当前探索是让自己的方法完成一次真实更新并获得可信执行证据，**不是宣布研究已经证明有效**。交接者应先推进这一主线，而不是退回旧缓存复现或以额外文档替代执行。
