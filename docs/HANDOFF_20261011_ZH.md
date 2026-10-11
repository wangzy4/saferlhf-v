# 项目进展与交接（2026-10-11）

> **核验快照：2026-10-11 14:36:56，UTC+8。** 本文是脱敏交接，不包含私有服务器地址、凭据、训练数据、权重或API原始回执。训练持续运行，步数只代表这个快照。
>
> **一句话结论：作者规模RM/CM已全部完成；全量RL数据准备、真实六角色EOS/保存恢复预检和最长样本容量预检均已通过；作者主线正式RL已从原始基座启动，确认32/5568次联合更新；最终64题生成和新Luna评测尚未开始。**

## 1. 用户授权与执行顺序

本轮授权为：**先准备、检查 → 按作者主实验复现强化学习 → 最后判官评测**。

- 只做LLaVA-1.5-7B的SafeRLHF-V主线，不声称复现整篇论文所有模型、baseline及基准。
- 新主线独立目录、独立记账；旧策略、旧评分器、旧主动实验和旧判官结果不覆盖、不追改。
- 使用原始LLaVA基座和新固定第三epoch的RM/CM；预检权重不warm-start正式训练。
- 不混入主动查询、ONS、LURE或主动算法的线性dual；不恢复已停止的Qwen任务。
- 固定正式第三epoch，不按验证结果或判官结果挑checkpoint。

## 2. 状态总表

|阶段|已核实状态|规模或验收|
|---|---|---|
|作者规模RM/CM|**完成**|各29,808训练对、590验证对、3epochs、5589 updates；全部验证独立重载差0|
|全量RL/PTX准备|**完成**|29,696唯一RL prompt；103,012条合格PTX池；无静默截断|
|源码/CPU门禁|**通过**|冻结源码hash核验；69项CPU测试通过；8rank launcher合同测试通过|
|六角色EOS/保存恢复预检|**通过**|2次保存的joint updates，三训练角色moments恢复，另1次未保存恢复后更新；fresh HF logits差0|
|最长真实prompt/PTX容量预检|**通过**|1次真实joint update；max512、context8192；保存及fresh HF reload差0|
|正式作者主线RL|**运行中**|2026-10-11 13:49:49进入正式阶段；14:36核实32/5568 updates，第一epoch|
|固定最终policy生成原64题|**未开始**|等待正式训练、固定epoch3和严格独立重载|
|新Luna配对评测|**未开始**|本轮新API请求0；本地监督器等待技术完成门禁；最多64次新请求|
|论文效果/主动算法优势/人类风险认证|**未证实或未做**|技术完成标记不代替效果或安全结论|

**旧文档中“真实预检尚未通过、正式RL尚未开始”的13:39快照已过时，以本交接为准。**

## 3. 已完成的作者规模评分器

两个独立7B评分器均从基座训练，language/projector/scalar head更新、vision冻结，固定第三epoch。29,824原始训练对中排除2行精确evaluation图像重叠、14行全局batch余数，最终29,808对；未复制样本凑30K。两角色各1863 updates/epoch、5589总updates。

|固定最终指标|RM|CM|
|---|---:|---:|
|官方590对严格排序|483/590 = **81.86%**|469/590 = **79.49%**|
|全训练对排序|87.16%|92.64%|
|train−validation gap|5.29个百分点|13.15个百分点|
|完整590对独立重载最大差|0|0|

CM零阈值sign **86.19%**、balanced accuracy **86.49%**、AUC **0.94210**。有害439条中漏检54，安全741条中误报109。回答pair内相关，不把1180条回答当作完全独立实验。

- RM epoch1排序83.22%高于固定epoch3；CM epoch3排序提高，但sign/loss较epoch2恶化。没有回滚选“最好”epoch。
- 论文30K参考为RM86.3%、CM80.4%、sign89.7%；作者划分/world size/完整launch未知，**尚未达到论文参考性能，不是严格同条件差值归因**。
- 发布偏好/安全标签主要来自GPT-4o，图像分类另含专家标注；不是全部直接真人回答偏好。
- optimizer抽样恢复通过，不等于完整cursor/RNG续训认证；CM末端LR=0时恢复后step不能证明非零参数变化。

详见 [作者规模评分器最终报告](AUTHOR_SCALE_RM_CM_ZH.md)。

## 4. 新RL数据口径与执行合同

### 数据

- 原始RL来源29,824行；排除114条与官方evaluation完全相同的题目、2条精确图像重叠、12条全局batch余数，最终 **29,696唯一prompt**。
- world8 × 每卡2 = global16；**1856 joint updates/epoch，3epochs共5568**。不要套用评分器的29,808对/5589步。
- 全部13个发布PTX shards，固定revision `788ba84f64b67c15fbc07bdf11ae113c3a19967b`；来源及合格池均 **103,012条**，内部重复保留，唯一image/question为89,714。
- 最大RL prompt706、最大PTX完整序列7250 tokens；369次实际processor token-ID抽查一致。context8192、生成上限512，无静默截断。
- RL/evaluation精确图像和精确题目重叠0，PTX与RL/evaluation精确图像重叠0；不是semantic/近似排重认证。
- 每个RL batch匹配一个独立shuffle的PTX batch，**全来源池不等于完整PTX epoch**；最终单列实际draws及unique覆盖。
- 实际shuffle seed：RL为42+epoch，PTX为100042+epoch。准备manifest将PTX种子文字描述为42+epoch，已记录说明性勘误；原manifest、parquet和源码冻结保持不改。

### 主线算法与工程配置

六个独立角色为actor、reference、reward critic、cost critic、RM、CM；只有actor和两个critic训练，vision及其余三个角色冻结。

|项|实际合同|
|---|---|
|方法|PPO、双critic、reward负KL/cost正KL、有界log-space拉格朗日、PTX|
|参数|KL0.02、PPO clip0.2、score clip50、value clip5、PTX16、gamma1、GAE0.95|
|dual|cost阈值−0.5、lambda初值10/上限20、SGD LR0.1、每rank滚动128成本窗口|
|优化|LR5e−7、betas(0.9,0.95)、wd0；actor cosine/3%warmup，critic constant|
|更新|actor用RL与PTX两次累积合成一次optimizer step；两个critic各一次|
|训练采样|temperature1、top_p1、top_k50、repetition penalty1、max512；不是主动实验的top_k0|
|执行替代|8卡、CPUAdam、ZeRO-3、BF16、SDPA、non-reentrant checkpointing|
|保存/选择|每epoch三训练角色checkpoint；每1000步actor HF snapshot；固定epoch3供最终评测|

作者实际world size/完整启动配置未知，上述执行替代明确披露，不能称原机原launch逐项恢复。critic实际constant scheduler不随actor warmup爬升；私有contract字符串“constant +3% warmup”须按实际scheduler/日志解释，后续勿据字符串误报。

## 5. 真实预检与失败修复

### 保留的首次启动失败

v1在模型加载前因torchrun将训练参数`--run`误作`--run-path`缩写而退出，**0参数更新**。旧失败标记、日志和正确停止的v1本地监督器均保留。v2显式加入`--`结束torchrun参数解析；8rank无模型测试复现旧错误并验证修复，之后才做真实预检。没有编辑运行中的冻结源码或覆盖失败证据。

### v2真实EOS/保存恢复预检

- 32 prompts，2次保存的joint updates；actor及双critic均真实更新、梯度探针有限。
- 三角色checkpoint的抽样moments恢复通过；恢复后另执行1次真实joint update，**该探针不保存进正式初始权重**。
- 独立stock-HF重载最大logit差0；actor控制logit最大变化0.15625；冻结角色/vision的hash不变。
- 总耗时约345.96秒。forced variable EOS只用于掩码/边界合同测试，不用于质量或正式训练吞吐结论。

### v2最长真实样本容量预检

- 最长16个RL prompt、最长16条实际PTX参与1次更新；PTX预检池32条，但本次draws/unique均为16，不能称32条全消费。
- 正式max512、无强制EOS；16个回答长度33–482，均EOS结束，0个撞生成上限。
- 三角色保存、冻结审计及独立stock-HF重载通过，logit差0；actor最大控制变化0.15625。
- 总耗时约176.29秒；报告的各rank峰值allocated最高36.06GiB、reserved最高39.74GiB，只是该预检，不是正式全程上限。
- 两次预检完成后重新核ext4真实可用空间944.33GiB及8卡空闲，再从基座独立初始化正式训练。

保留的日志包括DeepSpeed trace-cache失效提示及恢复探针的`c10d::broadcast_` autograd warning。门禁通过不构成所有分布式梯度数学等价的独立oracle证明，不隐藏这些诊断。

## 6. 正式训练快照与下一位接手的工作

截至14:36:56：phase为`AUTHOR_MAIN_SAFERLHFV_3EPOCH_FROM_BASE`；正式history有**32条完整joint update记录**、第一epoch；`DONE`及`RL_AND_GENERATION_DONE-v2`均不存在，未观察v2失败标记。本地仅一个v2监督器在等待，未启动新judge。

第32步记录lambda约20，已到预设上限；当前batch原始cost约0.217，而阈值为−0.5。**这是早期约束压力信号，不是安全合格，也不能据不同batch的cost拼出改善学习曲线。** 第32步actor实际LR约9.28e−8，仍处warmup；首次actor LR为0，因此optimizer step计数不等于每步必有非零参数变化。

### 接手优先级

1. **只监控已运行的v2，不重复启动**评分器、数据准备、预检或正式训练。重新核进程身份，不凭历史PID杀进程。
2. 查看phase、完整trace、history、v2失败标记及真实ext4空间；失败即保留证据并阻止生成/API，不用历史v1失败误判v2。
3. 跟踪三角色global steps、finite gradients/loss、frozen hash、长度和dual；每三角色checkpoint前必须满足250GiB真实空间门禁。GPU空闲是阶段快照，不是永久预约，不接管他人任务。
4. 正式结束核实5568计划更新是否全部完成、固定epoch3、实际PTX覆盖、冻结审计和strict fresh HF reload差0。**抽样optimizer/RNG/cursor保存不是完整resume认证**，当前无已认证的完整resume合同。
5. 生成固定最终policy的原冻结64题，greedy/max512，核ID/order/question/image hash、回答hash与metadata；server训练/生成外部API=0。
6. 只有`RL_AND_GENERATION_DONE-v2`和全部技术合同通过后，本地监督器才能发送新增64配对；不提前评测或按评测选checkpoint。
7. 更新本交接及正式结果报告，分别报告工程闭环、效果、裁判误差和未完成的人类认证。

### 识别v2产物（仅相对名称，不公开私有根目录）

- 正式run：`author-rl-20cat-3ep-s42-w8-zero3-v2`。
- 预检run：`author-rl-six-role-eos-preflight-v2`、`author-rl-longest-capacity-preflight-v2`。
- 最终生成run：`author-rl-official64-generation-v2`。
- 新失败：`RL_PIPELINE_FAILED-v2.json`；历史`RL_PIPELINE_FAILED-v1.json`必须保留。
- 训练/生成完成门禁：`RL_AND_GENERATION_DONE-v2`；本地监控：`SUPERVISOR_PROGRESS.json`。
- 正式训练源码仍冻结在 **`536732983716414cd89d79ce22a192de901f5b55`**；后续公共文档commit不改变运行源码。私有runner及连接信息只在私有交接中，本文不是可直接执行的服务器启动脚本。

## 7. 最终判官预算与不可越界事项

- 外传仅原冻结64题、对应原生图片和本次固定最终回答，配对已核验的base-before；**最多64次新Luna请求，每次联合helpfulness/safety两维**。
- 使用论文`2503.17682v2` Appendix C原prompt、iiot `gpt-5.6-luna`；是替代判官，不是论文GPT-4o逐数值复现。
- 不扩大至590题，不上传训练样本/权重，不自动API重试、图像fallback、输出repair或失败补判，不复用旧评分填新结果。
- 输出有效/失败/未尝试、主榜、分差、同文本敏感性、完整64缺失边界及usage。无可信billing时不猜人民币实扣。
- 原590/64已经使用，**不是新盲测**；旧Luna同文本强制偏好异常必须继续单列。

## 8. 历史工作保留与结论边界

- 旧小规模RM/CM、两组pilot RL、四策略各64题生成已完成；旧Luna主账本**379有效/5失败**，离线parser **380/4**只是另一账本，不覆盖主榜。
- 旧base后vs前helpfulness29/64、safety36/64；safe后vs前helpfulness29/62、safety26/62。旧工程闭环成功，但训练收益没有可靠复现，safe线存在退化警报。
- 自有主动实验 **N32/b8/T4/seed11** 完成：128候选、32新增AI proxy labels、actor及双critic各4次更新，exact全参数几何、回放差0、末policy严格重载差0及64题生成完成。**没有人类cert/ref标签，NOT_CERTIFIED，主动算法效果/公平标签效率未证实**；本轮不扩大该实验。
- 两个Qwen评测保持`STOPPED_BY_USER`，不恢复、不补判。旧active结果与新作者RL独立记账。

**最终判断必须分开：数学/工程闭环 ≠ 作者方法与规模复现 ≠ 论文效果复现 ≠ 自有算法优势 ≠ 人类安全认证。RM/CM准确率、训练cost或技术DONE都不能跨级替代最后一项。**

## 9. 关联资料

- [作者规模RM/CM最终报告](AUTHOR_SCALE_RM_CM_ZH.md)
- [项目进度报告](PROGRESS_REPORT_ZH.md)
- [历史综合结果分析](RESULTS_ANALYSIS_20261010_ZH.md)
- [旧Luna配对评测与审计](LUNA_AUTHOR_PK64_20261010_ZH.md)
- [自有主动N32实验报告](ACTIVE_SAFETY_V2_N32_ZH.md)
- [主动数学及指标边界](ACTIVE_SAFETY_METRICS_ZH.md)
- [主动v2初次工程验收](ACTIVE_SAFETY_V2_DEMO_ZH.md)
