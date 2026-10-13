# Codex 接续交接：Joint-Gradient / LLaVA-1.5-7B（2026-10-13）

> 状态截止：2026-10-13 18:22（UTC+8）。面向负责接续实现、验证和实验执行的 Codex；面向负责人的摘要见 [当前进展](../PROGRESS_20261013_ZH.md)。
>
> **这是公开脱敏交接，不是私有执行包。** 新增 Joint-Gradient 实现、测试、权重清单、原始回答、账本和证据目前位于私有工作区，未随本次文档提交公开。下文文件名是私有证据的定位键，不代表它们存在于本仓库。只有 GitHub checkout 时，先通过操作方批准的私有渠道取得工作区与环境映射；不得猜机器地址、复制凭据、上传私有包或宣称已经复跑。

## 1. 先恢复正确状态，不要回退到旧断点

- 主线为冻结 v3 设计下的 **Joint-Gradient / LLaVA-1.5-7B**，不是恢复旧 PPO step54，也不是切换 GRPO。
- 已有 **192 条真实新回答、四评分器各 192×2 次只读评分**；不是“等待首次生成”。
- 真实新增 Sol 调用和标签均为 **0**，本轮策略更新、评分器重训均为 **0**。
- 最新 Python CPU 回归为 **V9 / 129 项通过**；另外 JavaScript 源码隔离测试 **8 项通过**。V6 / 99 项已是历史版本。
- `training_preflight.py` 的 **CLI 已默认 Sol**；程序调用 `preflight(..., judge='luna')` 保留历史默认，需要显式 `judge='sol'` 才执行新协议检查。
- 单次发送器和真实账本标签接入代码已实现；不是“尚无 sender”。但它们仍未进行真实付费服务联调。
- 旧私有交接中“仅 10 个危险候选”“GPU 全占用而全项目等待”“sender 未实现”“preflight CLI 仍固定 Luna”等，是早期快照，不能当当前状态。
- 某条路线被门禁阻断时，继续独立、已授权的本地工作；不能因此停止整个项目，也不能绕过该门禁。

## 2. 不可越过的执行边界

1. **正式 RL 前先汇报**真实 A/B、四组 scorer 的质量比较和拟采用共同 recipe。推理与工程准备不等于策略训练许可。
2. 不自动 C/D、不重训 scorer、不扩大训练反馈，不恢复已停止的小模型支线，不扩大清理范围。
3. 所有真实外部请求必须走当前独立 Sol campaign 的累计次数与金额账本。费用额度、账本绝对位置和授权材料只从私有记录取得，不能每次新建账本重置预算。
4. 零自动重试。失败、超时、pending/崩溃都消耗预算；重复 request ID 不重发。保留失败，不修补标签、改协议或换模型冒充原成功。
5. GPU 每次启动 fresh 检查并取协作锁；无 compute 进程、显存占用符合入口条件且主机内存足够才启动。0% 利用率或剩余显存不等于独占许可。协作锁也不等于调度器 lease。
6. 不抢卡、不共享、不停止他人作业，不读他人训练日志或完整 argv/env，不留自动抢卡/自动续跑守护进程。
7. 新产物独占创建、私有权限，改代码前留版本，跨机迁移后校验 hash；原始证据、失败和旧协议不可覆写。
8. 不把合成测试、旧缓存 pre-A、原始评分、token usage、`image_used` 或 returned model 字段，升级成真实 A/B、质量收益、账单或底层视觉处理证明。

## 3. 阶段与数据合同

```text
独立协议试标 4
    → 共同启动反馈 64（映射 / 特征 / 校准）
    → 开发反馈 128（四 scorer 质量 / 真实梯度 A）
    → A 通过
    → 留置机制反馈 256 / B
    → B + 生产工程 + 训练前汇报通过
    → C / D（另有在线反馈、配方、长度和种子门禁）
```

准备预算设计为 452 次，不自动授权 C/D 在线反馈。pilot 标签不能暗中替代 64 条 startup 标签。现已离线准备的是 **4 + 192** 个 body，不是全部 452 个。

### 输入冻结与未完成接受

- 首批 192：startup64 / development128、危险意图候选96 / 良性候选96；机制256留置且未生成。
- 来源筛查后的候选池为危险来源819、良性239，候选数不等于最终接受数。
- v2 输入计划中的 44 条替换发生在生成和评分之前；现在不能根据已见输出、raw scorer 或截断情况重新择样、重生成或默默改 split。
- canonical image identity 为 `SHA256(b"RGB:<width>:<height>:" + RGB_pixels)`。旧 `str(rgb.size)` 前缀不兼容；旧 v3 危险候选的“无重叠”结论已撤销并留证。
- exact identity、来源链、pHash+dHash 是不同范围的完整性/操作性筛查，不是完整语义或 family 排除证明。大图列未全部独立解码的限制必须保留。
- 直接图文审阅未完成：当前会话图片读取工具明确不支持图像。不能以 caption、Text_only、CPU 解码或生成模型处理过图片替代审阅。
- 良性上游 family、许可和跨角色隔离仍未完成接受。MM-SafetyBench 已用于开发准备，不能再作为独立最终测试；final test 尚未接受。

### 生成与评分

- 本地准备 recipe：batch1、seed 为 `20261013 + 输入顺序`、temperature1 / top_p1 / top_k0 / repetition_penalty1、全 softmax 采样、max_new_tokens256、context2048。
- 显式新建 GenerationConfig，避免继承隐含生成默认值；这不是正式 RL recipe。
- 192 条共 28,350 个 response tokens，137 条 EOS、55 条到 cap 且无 EOS；无空文本，全部保留。
- 四 scorer 使用原始生成 token，`exact_generated_tokens_no_retokenization_no_added_EOS`，末有效 token 评分。
- L/H 为训练数据量档位：1K/3epoch 与约30K/3epoch；组合为 L/L、H/L、L/H、H/H。没有按结果选赢家。
- 四流重复差全0。RM low/high Spearman 0.383494；CM 0.733938。raw 尺度、相关性和零阈值符号差不能替代质量比较。

## 4. 实现地图：哪些是代码，哪些已在真实数据上运行

| 私有模块 | 当前能力 | 验证边界 |
| --- | --- | --- |
| `joint_gradient.py` | 一般八状态矩、ONS、water-filling、beta/KKT、查询、风险权重与 replay/LURE | CPU 数学与合成验证，不是生产 trainer |
| `ppo_objective.py` | response-only GAE、reward-only KL、共同 PPO、预冻 auto gate、八状态及式(13) scalar loss | 不以两个标量的无偏性代替真实 PPO 梯度校正 |
| `training_adapter.py` | 固定参数点逐样本 backward、事件系数、一次有效更新的适配接口 | CPU 等价性不认证 BF16/ZeRO/分布式更新 |
| `replay_b.py` | 多方法回放、独立 query/evaluation 几何、风险统计与 pool bootstrap | 真实机制 B 未执行 |
| `scorer_quality.py` | startup-only isotonic 映射、held-out 质量指标 | 当前没有真实标签可拟合 |
| `local_inference_v1.py` | 固定资产/图片/token 绑定的 prepare / generate / score | 192 新回答与四评分器真实运行完成 |
| `compare_fixed_scores_v1.py` | 四评分流顺序、回答 hash、revision、重复性校验及 raw 诊断 | 已在真实192条上完成；不选质量赢家 |
| `annotation_sol_protocol.py` | 精确 Sol 模型、单回答 helpfulness0–3 / unsafe0–1、真实 PNG/JPEG 输入 | 旧 Luna 协议独立保留 |
| `annotation_sol_budget.py` | SQLite 同一事务持久预留次数和微美元 | 费用上界须经审阅；unknown pricing 阻断 |
| `annotation_sol_dispatch.py` | reserve-before-transport、不可变 wire bytes、互斥单次调用、失败归档 | 故障注入测试通过，无真实外部调用 |
| `send_sol_once_v1.py` / `sol_responses_once_v1.mjs` | 显式单请求 CLI / SDK adapter；body与route绑定、零重试、禁重定向、响应大小限制、脱敏错误 | Python mock 与实际 JS 源码隔离测试通过；未加载真实 SDK 做服务请求 |
| `evaluate_four_scorers_v1.py` | 只读账本 join、拒 pilot/旧模型反馈、64拟合/128评估、四组合及4×32开发分组 | 合成测试通过；真实质量报告未生成 |
| `training_preflight.py` | A→B→C/D 累计门禁，固定 Sol 协议、预算证据，C/D 前汇报证据 | 只核查证据绑定和声明，不证明声明本身真实，也不启动训练 |

**尚未交付生产验收**：真实7B streamed scalar 等价、actor + 双 critic + dual + PTX 的一池更新、分布式 reduction/accumulation/ZeRO 尺度、严格 checkpoint continuation、最长样本容量、正式共同配方。

## 5. 数学与统计合同，不要偷偷换成旧损失

- 八状态为 `{0, 1/3, 2/3, 1} × {0, 1}`，helpfulness 除以3得到 R，unsafe直接为 C；旧连续严重度映射和旧成对标签不复用。
- 共同 PPO 优势为 `(A_R - nu*A_C)/(1+nu)`，clip 区间 `[0.8,1.2]`，response 长度平均，KL 仅进入 reward，auto gate 预冻。
- 旧原生 PPO、旧 active adapter 与 v3 的损失并不相同。不能直接复用旧 score Gram 并宣称等于实际 PPO 敏感梯度。
- 固定 actor / critic / nu / gate 参数点，反馈齐后计算事件系数，逐样本 backward，再加标签无关 PTX，最后一次有效 step。事件系数已含 pool 权重，不再次除 N 或 b。
- 7B 旧缓存 pre-A：72 次 loss-only backward、8组重复差0、参数/buffer hash不变；仿射误差0.79%–1.84%，使用 `GENERAL_EIGHT_STATE_EQ6`。未覆盖 clip-switch，不是真实新 A。
- B 的 query/evaluation 几何独立；beta 开始前冻结，风险统计不沿用帮助性 shrinkage；部署方法不可访问隐藏标签。
- B 设计为8个 N32池、预算2/4/8/16、每池500轨迹；轨迹只是池内 Monte Carlo，置信区间以独立 pool 为单位，不能当成数千独立样本。
- 四 scorer 的映射只拟合启动64；改变开发标签或分数不能改变映射。CM方向须独立核实，不能看到开发成绩后翻转。
- 当前质量数值筛查：RM 正相关且 MSE 优于 startup 常数；CM AUC≥0.7、balanced accuracy≥0.6、Brier优于 startup 常数。类别覆盖仍需审阅；数值过线不自动选赢家，也不替代真实梯度 A。

## 6. Sol 发送、预算与 preflight 接续注意事项

- 固定模型 ID：`gpt-6.1-sol`。旧 Luna protocol JSON、标签、回执和授权原义不变。
- body 为真实图片 + 原问题 + 一条真实回答；`store=false`、`max_output_tokens=1200`，不附 raw scorer、旧标签、来源路径或 caption 替代图。
- 金额使用十进制字符串并向上取整为微美元；已有花费上界、既有 pilot/main 次数都要计入。
- 已实现的费用依据只接受：经审阅、文件hash绑定、覆盖精确body及所有费用/失败结果、有效期内的单次全费用上界。**quota-only 路径未实现**；共享余额、告警和其他模型价格不合格。
- 单次 CLI 打开已有账本，不创建新 campaign，不自动 recovery/retry；reserve 成功不证明请求已上网。崩溃留下 pending 仍消耗预算。
- SDK 设置 `maxRetries=0`；内部180秒、子进程200秒超时；要求费用依据仍覆盖发送时间窗口。错误仅返回固定类别，不输出密钥或 provider 原始异常。
- 发送前还需当前许可、正确 route/transport hash、exact body allowlist、阶段接受条件；不要因为本地测试全部通过就执行收费 CLI。
- 四条离线 pilot 采用固定输入顺序前四条；**该位置策略是在生成后才记录**，不能声称生成前已预注册。pilot另占4次，不代替startup主反馈。
- Sol preflight 新增累计预算证据；C/D 另要求训练前汇报已经完整交付。所有历史 evidence 必须保持 protocol 和共同模型/映射/feature/critic bindings，不给旧 Luna evidence 换标签。
- **原 `PRETRAIN_PLAN_DRAFT.json` 仍是历史 Luna 草稿，不是可执行 Sol 计划。** 当前尚需独立创建真实 Sol 计划/证据连接，不能改写旧文件冒充过去已经通过。

## 7. 最新验证与定位键

以下均为私有文件名；不要把它们或内含数据直接提交公网。

| 定位键 | 核实结果 / 用途 |
| --- | --- |
| `PRETRAIN_CPU_VERIFICATION_V7.json` | 115项通过，Sol preflight + dispatcher故障测试 |
| `PRETRAIN_CPU_VERIFICATION_V8.json` | 121项通过，增加SDK adapter模拟测试 |
| `PRETRAIN_CPU_VERIFICATION_V9.json` | **129项通过**，13.926秒，RSS **580,354,048 B**；增加四scorer映射隔离与账本join测试 |
| `SOL_TRANSPORT_OFFLINE_VM_V1.json` | **8项通过**，RSS **62,189,568 B**；实际JS源码在模拟imports/网络/配置的VM内执行，未读取真实凭据、未加载真实SDK |
| `FOUR_SCORER_RAW_COMPARISON_V1.json` | 同192回答的四流真实raw诊断；`winner_selected=null` |
| `SOL_UNSENT_MANIFEST_V1.json` | 196条未发送body的冻结绑定，不是发送许可；本地清单不等于完整payload目录 |
| `LOCAL_GENERATION_SCORING_ARTIFACT_HASHES_V1.json` | 先前39个生成/评分产物绑定；不覆盖之后新写的全部源码 |
| `CODE_HASHES.json` | 原48项清单属于之前的同步批次；**新增sender/preflight/quality代码后尚需刷新**，不能称当前新增文件已被该旧清单覆盖 |

Python测试均 CUDA隐藏、单线程、RSS≤4GiB、180秒内部/195秒外部边界；这些回归 GPU/API/optimizer均0。不要把不同版本测试数相加当“不同用例总数”。

实际推理运行源码 `local_inference_v1.py` 的 SHA256 为：

```text
5142d5be59a9be09df08aa2d877ca5a135607652b06aff0a7d7093a1ae38ed0f
```

该绑定对应生成与四scorer共五次真实运行，不覆盖完整依赖图，也不认证未来训练配方。五模型内容revision、原图和回答的逐条hash从私有固定资产清单取得。

### 资源边界

- CPU准备与测试：单线程、RSS≤4GiB，有界JSON/Arrow；检查大image row group的压缩和未压缩大小，batch2不自动保证安全。
- 本轮GPU推理：单卡可见、无compute进程、使用显存≤100MiB、主机可用内存≥96GiB；GPU allocator40GiB、CPU RSS64GiB、内部1800秒/外部1830秒。
- 新生成实测：608.613秒、CPU RSS17.855GiB、GPU allocated13.633GiB。四scorer各约68秒、CPU约5.46GiB、GPU allocated13.474GiB。
- 旧缓存pre-A实测：934.970秒、CPU RSS52.222GiB、GPU allocated26.476GiB；不是4GiB CPU任务。
- 推理作业已结束并释放卡/锁，无自动续跑。之后的资源检查只作为历史快照，不证明现在仍有空卡。

## 8. 按这个顺序接续

### 可以立即继续的本地工作

1. 取得私有工作区映射后核对实际文件；保留旧 `CODE_HASHES`，补齐新增源码、测试与本轮产物绑定，再跨机校验。当前公开提交仅文档，不假装已发布可执行实现。
2. 更新私有交接的过时顶部状态；保留原始历史段落。另建 Sol draft/证据连接，保持真正未完成项为 blocked，不填合成 `PASSED`。
3. 根据现有实现继续输入/来源/许可审计，以及生产scalar、critic/dual/PTX、分布式尺度和恢复准备。任何GPU执行都先检查权限与资源；不得越过正式RL前汇报。
4. 使用既有有界runner重跑涉及修改的测试，保留新版本结果与失败；无需重新生成192回答或重训scorer。

### 只有相应门禁齐全才执行

5. 有效费用依据和当前发送条件齐全后，最多4次独立Sol协议试标；通过后再取得64启动+128开发真实主反馈。缺失/失败标签不补猜、不重试或用pilot顶替。
6. 完成图文/family/许可接受，并独立核实固定CM方向。用只读绑定账本接入 `evaluate_four_scorers_v1.py`；startup拟合、development评估，保留四组合，不自动挑赢家。
7. 冻结共同scorer、映射/特征/校准器及相应recipe，执行真实A；通过后推进留置256机制数据/B。
8. 正式RL前交付真实结果汇报。生产更新、strict resume、最长样本容量、独立final test、训练长度/种子和C/D反馈许可齐全后，才按共同训练量与共同新增反馈预算运行对照。

不要因第5–8步受阻而放弃第1–4步；也不要用第1–4步的工程成果替代第5–8步的真实证据。

## 9. 必须保留的历史边界

- 约30K RM/CM各29,808训练对、590验证对、3epochs、5,589更新已完成；现存两档scorer不需要为本轮比较重训。
- 旧八卡PPO停止于54步，没有可恢复正式checkpoint；六卡v4即时恢复通过但续训轨迹失败，v5两次fresh-run一致不等于strict checkpoint resume。
- 旧N32四轮/AI-proxy反馈不是v3启动反馈；旧N8和7B pre-A不是新A/B。旧Luna外评不是Sol标签。
- 旧已删除policy/失败全量快照不可恢复；缓存、日志、失败与审计仍保留。不扩大删除范围，也不把合成传输当真实完整模型云备份。
- 无新增真人金标，无新策略安全性—帮助性改善或主动反馈效率优势的认证。

历史详情见 [10月12日资产与恢复报告](../PROGRESS_20261012_ZH.md)、[约30K scorer报告](../AUTHOR_SCALE_RM_CM_ZH.md)。新状态以本页和本次私有验证记录为准，而非历史文档中的“当前/下一步”。
