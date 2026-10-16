# Joint-Gradient接续：真实一步、checkpoint及失败恢复

> 核对时间：2026-10-16 22:55（UTC+8）。本文件接替[21:06交接](CODEX_HANDOFF_20261016_ZH.md)的当前状态和下一步，不改写历史失败。仅公开脱敏文档；以下私有定位键不表示文件已发布到GitHub，不得据此猜测服务器、复制凭据或公开私有代码/数据/权重。

## 1. 首先准确报告三个计数

- **可用checkpoint的策略更新深度：1。** 成功执行actor/双critic各一步，dual完成，checkpoint已提交。
- **全部尝试累计实际actor step：2。** 失败执行一步未保存，恢复执行从原基线重新做一步；不是连续两步，不是从失败权重恢复。
- **本轮反馈：4次尝试、4次成功、0重试。** 历史196次成功不变；总成功200。校准器观测数为历史/启动192加本轮4，合计196。

不要重跑任何已预留/完成的更新，不要重置反馈账本，不要把生成旧概率的事后回放说成原logits存档。

## 2. 用户的新要求与持续执行

用户原话：

- “你的权限是比较大的，标注限定不要在意！！！”
- 结构化确认：“当前32及既有候选池均可按需标注”。
- 在得知actor真实一步后保存失败、需要恢复执行时：“直接做放开来往下推少要授权”。

因此，原4次上限不再是标注授权障碍。当前四查询估计器未修改，不为使用授权而盲目补标签。追加标注须版本化、保留训练/留置/测试隔离，不倒改历史选择。工程修复和该恢复执行已获准，不要反复问相同许可；也不能把这些话解释为自动通过正式C/D、数据接受或可以抢占资源。

私有授权追加记录：`OPERATOR_ANNOTATION_EXPANSION_20261016_V1.json`、`OPERATOR_RECOVERY_CONTINUE_20261016_V1.json`。原授权文件因冻结hash而保留，仅作为历史范围。

## 3. 已完成的主线

1. `real32-gradient-full-v3/`：1184次真实反传，完整一般八状态Gram，全部6,759,919,616可训练参数，actor不变，退出0；本地/远端SHA一致。
2. `real-one-pool-capacity-v1/`：失败，PTX loss漏传必填`reduction`，没有optimizer step；保留现场。
3. `real-one-pool-capacity-v2/`：通过。显式单回答`mean`、外层16/32，和固定逐回答均值目标一致；最长PTX上下文991，双critic最长fresh反传、三套CPU AdamW状态完成。
4. `fresh-feedback-v2/sequential-v1/`：4次真实顺序查询成功，ACTUAL_REVEALS和ALIGNED196_STATE已生成。发送端保持既有一次性SDK，凭据不转移到GPU环境。
5. `real-one-pool-update-v2/`：actor已step，权重hash改变；随后`save_pretrained`引入DeepSpeed时因`CUDA_HOME`缺失失败。两个critic未step，staging无权重，不可resume。
6. `real-one-pool-update-v3/`：修复launcher环境，使用既有toolkit，训练前执行真实保存/读回测试。从原基线恢复执行后，actor/双critic各一步，退出0，checkpoint提交。
7. `INDEPENDENT_POSTEXIT_FILE_VERIFY_V1.json`：退出后独立核对36个文件、244,899,215,386字节，全文件hash一致。
8. `checkpoint-reload-fixed32-v1/`：独立新进程加载三套真实模型，完整模型/buffer hash一致，每个模型对原32条固定回答forward，有限输出、权重不变；新增step/生成/反馈均0。

成功更新共32次PPO反传、32次PTX反传、每个critic32次反传。式(15)未裁剪风险估计0.2973857006268199，ν从1变为1.1973857006268198。它不是独立评测危险率。

固定训练回答token上的actor更新后/冻结FP32概率比率为0.95004930298至1.13133548590，仅用于确认权重变化后的可执行性，不证明安全/帮助性收益。

## 4. 冻结方法仍不变

继续遵守旧交接第7节：全八状态、全可训练坐标、固定token/gate、真实Gram选择；先合并优势再PPO clip；同actor点累计PPO/PTX并一次step；32维八头JointONS与1/(Mq)在线权重；式(15)按实际揭示顺序和未裁剪校正风险更新。没有仿射替代、sketch、降精度、重生成或按结果挑样。

v3相对v2仅改变恢复预留文件名，launcher补齐既有CUDA路径并强制保存预检；方法/输入/反馈不变。保留v1/v2源码及失败输出。不要把CPU测试中的tiny optimizer step计入真实7B策略更新数。

## 5. 私有接续定位键

| 定位键 | 用途 |
| --- | --- |
| `REAL32_GRAM_LOCAL_VERIFIED_20261016_V1.json` | 完整Gram与本地/远端hash验证 |
| `REAL_ONE_POOL_SOURCE_SHA256_V{1,2,3}.txt` | 原版、容量修复、保存修复源码绑定，旧版保留 |
| `REAL_TRIAL_FEEDBACK_V1.log/.exit` | 真实四查询完成 |
| `fresh-feedback-v2/campaign.sqlite` | 独立持久账本，禁止重置 |
| `fresh-feedback-v2/sequential-v1/ACTUAL_REVEALS.json` | 本轮真实揭示；GPU侧定位键`ACTUAL_REVEALS_SOL_V1.json` |
| `real-one-pool-update-v2/OPTIMIZER_0_STEP_COMPLETED.json` | 失败尝试确已执行actor一步的证据 |
| `real-one-pool-update-v2/FAILED.json` | 保存路径错误，无已提交checkpoint |
| `REAL_UPDATE_ATTEMPT_RESERVED_V1.json` | 原失败尝试预留，不能删除后重跑 |
| `REAL_UPDATE_RECOVERY_ATTEMPT_RESERVED_V3.json` | 经授权恢复执行的独立预留 |
| `real-one-pool-update-v3/DONE.json` | 成功pool和checkpoint提交 |
| `real-one-pool-update-v3/CHECKPOINT_{0,1,2}_READBACK.json` | 三套模型/优化器/scheduler精确读回 |
| `real-one-pool-update-v3/checkpoint/{FILES,TRAINING_STATE}.json` | 文件manifest、dual/cursor/状态绑定 |
| `real-one-pool-update-v3/INDEPENDENT_POSTEXIT_FILE_VERIFY_V1.json` | 进程退出后的全文件hash核对 |
| `checkpoint-reload-fixed32-v1/DONE.json` | 新进程真实模型加载及96个forward检查 |

完整模型和优化器文件仍只在私有训练工作区；本地同步的是具名JSON证据，不是完整checkpoint副本。

## 6. 下一步，不退回“训练尚未发生”

当前已越过首次真实一步的工程目标。继续处理checkpoint恢复验收及真实A→留置B→生产验收，不要将旧“零步/待授权/Gram仍在跑”作为现状。

- 已通过：持久文件hash、保存时逐张量读回、新进程模型/buffer重载和固定回答forward。
- 未通过：完整optimizer/scheduler/RNG/dual/cursor恢复后继续执行等价、分布式strict resume、正式A/B和生产验收。
- 训练输入fixed32诊断不是盲测。后续独立指标、方向和数据须在查看相应结果前固定；不选择性使用标签、不污染留置/最终测试。
- 原人工视觉/family/权利链/split缺口、RM附加校准失败、共同CM未选、正式C/D范围仍保留。
- 需要标注时按新授权推进，不以4次上限停工；失败/状态不明请求不得隐式重发。
- 每次GPU工作仍需实时空闲检查和协作锁，不查看或终止他人任务，不自动安装依赖。

阶段性报告要先给真实step、调用与checkpoint状态，然后继续具体工作。不要宣布研究有效，也不要用增加文档或测试数量替代剩余验收。
