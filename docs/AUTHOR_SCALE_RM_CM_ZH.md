# 作者规模 RM/CM 全参数训练与最终验收（2026-10-11）

> **2026-10-12更新：正式RM/CM结果仍有效，全部三epoch checkpoint和最终导出保留，不需重训。** 下文八卡RL运行与监督器描述为历史：正式run已停止于54步、无可恢复正式状态；六卡续训仍待验收，最终新生成与判分未开始。旧预检checkpoint/policy及失败v3/v4快照已按用户要求删除。当前资产与下一步门禁见[10月12日进展](PROGRESS_20261012_ZH.md)。

## 本轮授权与范围

用户先授权按作者规模重新训练RM/CM。新建独立目录，从固定基座分别初始化，不续接1K模型、不覆盖旧权重，不上传训练样本或模型。**CM和RM均已完成正式3epochs、各5589次更新，以及全部590对独立严格重载；2026-10-11 12:52（UTC+8）核验两角色及根完成标记。** 工程验收通过，不等于达到论文效果。

用户随后另行授权“先准备、检查，再作者主线RL，最后判官”。新的独立RL流水线已完成全量数据隔离及真实六角色/最长样本验收，再从原始基座训练固定3epochs，最后仅对原冻结64题做一个新配对、最多64次Luna请求。**截至2026-10-11 14:36（UTC+8）核验，正式RL已进入第一epoch并完成32/5568次联合更新，最终生成及新API判分尚未开始。** 这不追改旧结果，也不扩大主动标注实验或恢复Qwen。最新验收细节及接手任务见[交接文档](HANDOFF_20261011_ZH.md)。

## 作者的方法与可核验来源

- 论文 [2503.17682v2](https://arxiv.org/abs/2503.17682v2) §4.2、Appendix A.1：两个独立评分模型从 **LLaVA-1.5-7B** 初始化，完整图像＋问题＋回答的末有效token给出标量分数。
- **RM**：帮助性Bradley–Terry排序，`-log sigmoid(R_helpful-R_other)`，加分数平方正则。
- **CM**：让更有害回答的成本更高，安全偏好排序加两条回答的绝对安全监督，再加分数正则。公开代码绝对项使用 **`-harmless_rate`（−3至3的原标注取负）**；不是强制转换成±1，rating=0保持常数项/零梯度。CM不是Beaver-Guard-V多级分类模型。
- BeaverTails-V的回答偏好/安全评分主要由GPT-4o标注；图像分类还涉及专家标注。发布数据不等于全部直接真人回答偏好，不新增付费标注请求。
- 论文Table 4最大档为30K：RM排序86.3%、CM排序80.4%、CM sign89.7%。它们是参考值，不是我们的结果，也不能忽略未知划分/配置差异直接宣称逐数值复现。

## 数据规模与隔离

固定发布版本`ee19041205c720c0faea575de563de8a6a8f9094`的20类**train合计29,824对，evaluation合计590对**。本轮尽可能使用全部合格train，不复制样本凑30,000，不再以20类均衡小子集代替全量。

- 保留原发布偏好、安全标签和rating，不重标、不根据新模型评分过滤。
- 排除非法记录、训练同文本对、超过4096的完整序列以及与官方evaluation精确解码图像重叠的训练行；记录各原因和私有源ID，不静默截断。
- 同一split内重复图像可以对应不同问题/偏好对，保留；train/evaluation之间精确图像重叠为0。不是近似/语义排重认证。
- 训练数取全局batch16的整数倍，最多额外排除15行并明确计数；实际数量以准备后的manifest为准。
- 验证保留完整官方590行，不为多卡整除补样本或丢样本。官方evaluation及其中64题此前已用于诊断，**不是新盲测**。
- 旧小模型的train行可以重新进入这次从基座训练的大模型；旧内部验证不再用于宣称新模型独立泛化。固定官方验证，不选中间“最好”epoch。

## 本轮实际配置

|项目|设置与来源|
|---|---|
|基座|`llava-hf/llava-1.5-7b-hf`，固定revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`|
|训练参数|完整语言骨干、投影层、scalar head；视觉塔冻结；非LoRA|
|epochs / seed|3 / 42，公开RM/CM YAML|
|LR / Adam betas / wd|3e-5 / (0.9,0.95) / 0，公开YAML|
|scheduler|RM constant_with_warmup；CM cosine；3% warmup|
|regularization / CM scale|0.001 / 1，公开YAML|
|上下文 / 精度|4096，BF16，原生FP32 RoPE，SDPA|
|每卡batch / 累积|4对 / 1，公开YAML|
|执行|4张H20串行CM→RM，全局16对；ZeRO-2、FusedAdam、non-reentrant checkpointing|
|持久化|每1000 updates及每epoch存checkpoint，固定末轮模型；独立严格重载全部验证|

**边界：** 作者实际world size、数据清单和完整启动命令未知。公开示例只指一个类别且留下不少null参数；本轮明确补齐全20类数据。四卡/ZeRO-2、SDPA、无词表head梯度的显式冻结，以及`num_logits_to_keep=1`是我们的已验证执行路径，不冒称作者原机原配置。理论上断开的词表head在wd=0时本来也不会更新。

正式训练前使用最长真实样本单独做容量试验，两角色从基座重新初始化，不把预检权重续接到正式训练；发现问题保留失败证据、修正工程，不悄悄缩数据或修改损失。

## 规模化的工程修正与验收

- AUC改为排序/searchsorted的Mann–Whitney计算，tie仍为0.5，避免全训练集构造`N_unsafe×N_safe`矩阵；不改变优化损失。
- 多卡验证使用无补样本的rank-strided索引；独立重载按同一batch组成比较所有590对。
- 检查optimizer moments抽样恢复及后续optimizer step。CM cosine末端LR可能为0，记录LR，不把step计数说成必有非零参数变化。
- 独立加载前释放训练engine和optimizer，避免额外同时容纳两个7B模型及Adam。保存checkpoint包含每rank RNG及数据游标，但**并不等于完整断点续训已验收**，没有新`--resume`合同。
- 数据、缓存、checkpoint均写物理ext4，启动时重新检查GPU和空间；空闲是快照，不是预约。

## 什么才叫评分器“好”？

训练完成不自动保证质量。本报告按预先约定使用固定最后epoch，分别报告：

1. RM/CM严格排序准确率、ties、分类型结果，训练/验证gap及loss分项；
2. CM零阈值准确率、balanced accuracy、AUC、有害漏检/安全误报；明确多数类基线及全判安全塌缩；
3. score分布、损失与参数梯度有限性、全验证严格重载、保存/optimizer恢复合同。

独立生成场景的泛化、人类风险与新算法标签效率仍需要后续独立验收；使用新评分器接RL依据上述新增授权，不把本次排序/sign准确率当作policy安全认证，也不按判官分数选checkpoint。

## 完成经过与工程验收

状态已推进：全量数据准备完成，**29,808 train / 590 evaluation**，原始29,824训练行中排除2行精确evaluation图像重叠，另明确排除14行全局batch余数；没有非法标签、同文本训练对或长度超限排除。最长完整序列3146 tokens，188次实际processor token-ID抽查一致，无截断或重标。每个正式模型实际完成**5589次optimizer updates**（每epoch1863次）。

GPU预检首次因launcher遗漏`CUDA_HOME`在导入DeepSpeed时退出，**0次更新**。已保留失败、修正CUDA工具链环境并通过恢复流程；数据hash复核后复用，不重复准备。最新69项CPU回归（含严格FP32/BF16原生loss/梯度对照）通过。

- 小模型CM/RM均完成各2次更新、optimizer恢复探针及17对独立重载，分数差0。
- **真实LLaVA-7B最长样本CM/RM均完成各1次更新**，保存/恢复/17对独立重载通过，分数差0。每个容量run耗时约2.4分钟（另有launcher开销），峰值allocated约62.09GiB、reserved约64.96GiB；包含独立重载，不是全量训练全程峰值上限。预检权重不续接正式训练。
- CM于2026-10-10 15:53:58（UTC+8）进入正式阶段，RM于2026-10-11 02:13:41进入正式阶段；时间包含模型初始化/验证。各自总耗时约10.32h/10.37h，训练更新计时约9.33h/9.35h，不相加为并行训练时间。
- 正式训练每角色峰值allocated约62.09GiB、reserved约66.61GiB（含独立重载）；不把预检64.96GiB的reserved当作正式全程上限。
- 两个最终模型均完整590对独立重载，最大分数差**0**，排序保持；optimizer抽样moments恢复及恢复后step通过。RM恢复后LR=3e-5，CM为0；后者不证明恢复后有非零参数更新，也不构成完整cursor/RNG续训认证。

## 固定第三epoch的最终结果

排序采用严格`>`，ties保留且不计正确。两列不是生成回答PK胜率。

|指标|RM|CM|
|---|---:|---:|
|官方590对排序|483/590 = **81.86%**|469/590 = **79.49%**|
|验证ties|0|2|
|全29,808训练对排序|25,980/29,808 = 87.16%|27,614/29,808 = 92.64%|
|train−validation gap|5.29个百分点|13.15个百分点|
|验证loss|0.436733|1.257716|
|独立全验证重载max score diff|0|0|

CM对1180条回答的零阈值sign准确率**86.19%**，balanced accuracy **86.49%**，AUC **0.94210**。多数类/全部判安全基线62.80%；并非全判安全塌缩。有害439条：TP385、漏检54（12.30%）；安全741条：TN632、误报109（14.71%）。同一pair的两回答相关，不能把1180当作完全独立实验，也不是人类风险置信认证。

### 每epoch验证轨迹（不据此选模型）

|epoch|RM排序|RM loss|CM排序|CM sign|CM loss|
|---|---:|---:|---:|---:|---:|
|初始化|42.54%|0.761951|49.66%|56.36%|2.461850|
|1|83.22%|0.427609|75.08%|85.34%|1.370135|
|2|82.20%|0.428953|77.46%|87.03%|1.161000|
|**3，固定使用**|**81.86%**|**0.436733**|**79.49%**|**86.19%**|**1.257716**|

RM的epoch1排序高于固定epoch3；CM的epoch3排序提高，但sign和loss较epoch2恶化，安全误报99→109，有害漏检仍54。**继续训练不是所有质量维度单调改善**；CM较大train–val gap也是泛化警报。不回滚或挑选“最好”epoch。

### 最终分类别排序

每类仅28–30对，以下是描述性结果，不做显著性或类别优劣认证。

|类别|n|RM正确|CM正确|
|---|---:|---:|---:|
|animal_abuse|30|24|25|
|dangerous_behavior|30|24|27|
|deception_in_personal_relationships|30|27|26|
|discriminatory_depictions|30|26|21|
|environmental_damage|28|25|16|
|false_information|30|28|21|
|financial_and_academic_fraud|29|23|25|
|hacking_or_digital_crime|28|24|23|
|harmful_health_content|30|25|25|
|horror_and_gore|30|21|20|
|identity_misuse_and_impersonation|29|26|27|
|insulting_and_harassing_behavior|30|23|25|
|pornographic_content|30|24|23|
|privacy_invasion_and_surveillance|30|25|24|
|psychological_harm_and_manipulation|30|21|23|
|psychological_horror_and_dark_themes|29|25|22|
|sensitive_information_in_key_areas|29|26|24|
|sexual_crimes|30|20|24|
|terrorism_or_extremism|29|23|22|
|violence_and_physical_harm|29|23|26|

### 分数与损失诊断

最终验证RM score均值−0.2108、标准差1.8462；CM均值−0.9347、标准差4.1985，有害/安全均成本3.1144/−3.3335。RM pairwise/regularization loss为0.433281/0.003453；CM pairwise/absolute/regularization为0.459581/0.779635/0.018501。验证诊断重新用FP32算，与训练启用的原生BF16运算及聚合顺序分开；这里的梯度诊断仅针对分数，不冒称全网络梯度oracle。这些分数及sigmoid不是校准的人类有害概率。

## 结论与后续边界

**作者规模评分器工程闭环完成，有可见排序和安全标签泛化信号，但论文性能尚未复现。** 对论文30K参考档，RM81.86% vs86.3%、CM79.49% vs80.4%、sign86.19% vs89.7%；差值不能剥离未知作者split/world size/启动参数来作严格同条件结论。一个seed、已使用验证集、AI来源标签及类别小样本限制仍在。

新作者RL仅用原始LLaVA基座和本次固定新RM/CM；PPO、双critic、原生KL、PTX及有界拉格朗日保留，不混主动查询或ONS。完整发布PTX的103,012条全部成为合格池；RL从29,824原始prompt中排114条与官方evaluation完全相同的题目、2条精确图像重叠和12条全局batch余数，实际**29,696唯一prompt**、每epoch1856/三epoch计划**5568次联合更新**。RL/evaluation图像和精确题目重叠为0，PTX与RL/evaluation图像重叠为0；PTX内部保留重复，唯一image/question为89,714。最大RL prompt706、最大PTX序列7250 tokens，369次实际processor IDs抽查一致，无截断。每次RL更新匹配一个独立shuffle的PTX batch，**全来源不等于完整PTX epoch**，实际覆盖待运行统计。8卡/ZeRO-3/CPUAdam/SDPA是明确工程替代；评分器的29,808对/5589步不是RL的prompt数/更新数。

首次RL预检launcher因`--run`被torchrun误作`--run-path`缩写，在模型加载前退出，0参数更新；日志和失败标记保留。新独立版本加显式参数分隔符`--`，8rank参数转发测试通过，再做真实预检。**最新已通过六角色2次保存更新、三角色moments恢复和未保存恢复后更新，以及最长真实样本1次max512更新；两次fresh HF reload差均为0、冻结审计通过。** 正式训练从基座重新初始化，不沿用预检权重。监督器对旧失败正确停止；新监督器等待正式训练和64题生成门禁，API尚未触发。

最终Luna仅原64题与旧base回答比较，原Appendix C、原生图像、零自动重试；Luna替代论文GPT-4o、64题不是新盲测，已知同文本强制偏好异常需单列敏感性。所有新结果独立记账，技术DONE不等于安全质量合格或人类认证。

另查明历史helper的CM绝对安全项采用FP32，而上游整数rating与BF16分数保留BF16运算。新运行显式启用`--native-loss-precision`，同时对齐图与运算创建顺序；旧默认保留，不静默改历史模型/指标。这是已验证的有限精度差异，**不是已确认的旧RL退化根因**。

期间保留 [1K历史结果](RESULTS_RM_CM_ZH.md)、[综合结果分析](RESULTS_ANALYSIS_20261010_ZH.md) 和原失败，不追改历史账本。
