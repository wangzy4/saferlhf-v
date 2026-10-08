# Safe RLHF-V 复现与改进路线

## 状态与来源

本仓库 fork 自 https://github.com/saferlhf-v/saferlhf-v ，基线提交 `337d200192e6d0a3c3d61c389f38f964bd2288e3`。当前 bootstrap 分支只增加复现文档和审计工具，不修改训练算法；尚未完成环境安装、权重推理或训练验证。

- 论文：[Safe RLHF-V: Safe Reinforcement Learning from Multi-modal Human Feedback](https://arxiv.org/abs/2503.17682)，调研版本 v2。
- 数据：[BeaverTails-V](https://huggingface.co/datasets/saferlhf-v/BeaverTails-V)，调研 revision `ee19041205c720c0faea575de563de8a6a8f9094`。
- 策略权重：[SafeRLHF-V](https://huggingface.co/saferlhf-v/SafeRLHF-V)，调研 revision `9fa9092c3d09484039e299cb92dcf9fe945075fa`。
- 框架来源：[align-anything](https://github.com/PKU-Alignment/align-anything)。不要在未核对版本的情况下直接替换为最新框架。

## 论文要点

1. **BeaverTails-V**：对 helpfulness 和 harmlessness 分别标注偏好，包含安全程度分级。论文分类为 9 个一级、20 个二级类别；HF 当前发布 20 个 config，总计 **29,824 train + 590 evaluation**，元数据声明下载大小约 **3.86 GB**。不要把论文收集阶段的 32k prompts 当作当前训练集大小。
2. **Beaver-Guard-V**：对输入和输出进行安全判别，并通过过滤、重新生成降低攻击成功率。它是论文的另一个实验方向，不等同于策略的约束 RL 训练；本仓库未提供完整独立 guard 训练/评测管线，HF 当前账号下也未找到其独立权重。
3. **Safe RLHF-V**：独立训练 helpfulness RM 和 harm CM，再以 PPO 更新策略及 reward/cost critics。通过动态拉格朗日乘子和 budget bound 控制安全约束，不只是固定的 `reward - alpha * cost`。
4. 论文摘要报告安全与帮助性分别提升 34.2% 和 34.3%，这是作者结果，尚未验证。主表使用 **GPT-4o 对基础模型的双维度 pairwise win rate**，不是简单安全分类准确率；评测覆盖 BeaverTails-V、MM-SafetyBench、SPA-VL、VLGuard、VLSBench，评判提示在论文 Appendix C。

## 已发布资产与复现边界

HF 模型仓库包含三个子目录，而非可直接统一加载的单个根目录：

- `LLaVA_Safe_RLHF-V`
- `LLaVA-NeXT_Safe_RLHF-V`
- `Qwen_Safe_RLHF-V`

各目录包含 config/tokenizer/processor 与 `pytorch_model.bin`。推荐按子目录下载并使用本地路径，先做权重完整性、模型类型、chat template 和生成验证。发布格式为 pickle `.bin`，应仅加载可信来源并记录校验和与框架版本。

**未找到公开 RM/CM checkpoint**：上游 [issue #7](https://github.com/saferlhf-v/saferlhf-v/issues/7) 也在询问，当前无回复。最终策略权重可用于推理复现，但不能代替 RL 阶段需要的 RM/CM，应准备自行训练。上游 [issue #1](https://github.com/saferlhf-v/saferlhf-v/issues/1) 有 Qwen 权重生成重复的反馈，需单独测量重复率与解码参数。

代码 Apache-2.0；数据为 **CC-BY-NC-4.0**，不能因为代码许可证宽松就默认训练数据和衍生用途允许商业使用。模型许可证需另外核查。

## 静态审计：先解决这些问题

### 已确认的参数解析错误

`trainers/text_image_to_text/{rm,cm,dpo,ppo,safe_rlhf_v}.py` 的 main 使用：

```python
keys = [k[2:] for k in unparsed_args[1::2]]
values = list(unparsed_args[2::2])
```

但 `parse_known_args()` 返回的未知参数不包含程序名。输入 `['--actor_model_name_or_path', 'base', '--epochs', '3']` 会得到错误的 key `se`，丢失首个 flag，并错配后续参数。应首先统一实现严格的 key/value 解析，测试空参数、单项、多项、负数、缺失值和未知 key，再逐一修复所有入口；不能直接运行 README 的脚本并相信 CLI 参数生效。

### 需要运行验证/进一步修正

- `pyproject.toml` 依赖几乎未锁定。vLLM、PyTorch、Transformers、DeepSpeed 的最新组合未必兼容旧模型代码；依赖清单未显式声明代码直接导入的 PyYAML。先用独立环境锁定依赖，vLLM 非必要时避免与训练环境耦合。
- YAML 没有 `processor_kwargs`/`lora_cfgs`/`bnb_cfgs`。**不是简单的缺字段 AttributeError**：`dict_to_namedtuple` 对缺字段返回 None，`namedtuple_to_dict(None)` 返回 `{}`。需检查每个消费点，不应盲目添加配置并声称修好了。
- `SafeRLHFVTrainer` 未调用基类构造函数；基类的 `self.lora_cfgs`、`self.bnb_cfgs` 初始化路径需检查，特别是保存/LoRA 分支。
- `safe_rlhf_v.py` 在 actor 与 **cost** tokenizer 相同时却赋值 `self.reward_tokenizer = self.tokenizer`，疑似变量笔误；后续又无条件替换 cost tokenizer，需要同/不同 tokenizer 的测试。
- YAML 有 `cost_critic_model_name_or_path`，但代码实际从 `cost_model_name_or_path` 加载 cost critic。应明确支持独立 critic checkpoint 或移除误导性配置。
- CM 模板有意把 safer response 放到低 cost 一侧，且 harmless rate 取负。不能只凭变量名 `better`/`worse` 就翻转顺序；应验证“有害回答 cost 更高、安全回答 cost 更低”、0 分标签和 7 档评分对应的损失。
- 示例只训练 `animal_abuse`，不是全 20 类。完整训练要明确合并各类别、类别权重和 train/evaluation 隔离，避免同图不同问答泄漏。
- 默认 RL 序列长 8192、生成 512；需先用小 batch/短序列测显存，不能假设 8 卡一定直接跑通。
- `scripts/setup.sh` 生成 `MASTER_PORT`，但需要验证实际 launcher 使用该端口；脚本依赖从 `scripts/` 启动，后续应改成相对脚本路径定位。

## 推荐复现顺序（先 LLaVA-1.5-7B）

### P0：环境与数据审计

- 独立 Python 3.11 环境，记录驱动、torch/CUDA wheel、Transformers、DeepSpeed、datasets 等精确版本。驱动显示的 CUDA 13.0 是兼容能力，不代表已安装 CUDA toolkit；DeepSpeed 编译需要匹配的 toolkit/nvcc。
- 修复参数入口，建立单元测试；安装后验证 imports、模型加载、processor/chat template、单卡 forward。
- 固定数据 revision，检查 ID/评分类型、平局/无效偏好、图像解码、跨 split 重复；建立训练内验证集，公开 evaluation 留作最终测试。
- 缓存与 checkpoint 放大容量数据盘，不提交到 GitHub。

### P1：推理复现（优先得到可信结果）

- 基础 `llava-hf/llava-1.5-7b-hf` 与发布策略，在同一批 BeaverTails-V evaluation 上推理。
- 固定 prompt、chat template、max_new_tokens、seed 与解码参数，保存逐样本输出和元信息。
- 按论文 Appendix C 用 GPT-4o 对 helpfulness/safety 分开 judge；随机交换 A/B 并记录 ties、失败与重复率。外部 API 涉及数据发送与费用，启动前确认授权；记录准确 judge 版本。
- 本地 guard 评估可补充，但不应冒充论文 GPT-4o win rate。

### P2：最小训练闭环

- 先单类、128–512 对样本做 RM/CM smoke test：验证 pairwise loss、排名准确率、cost 符号及安全分类。
- 从 RM/CM 初始化 critics，用 32–128 prompts 做几十步 Safe RLHF-V，先关闭 PTX 简化调试，再开启。默认参数仅作起点，不宣称精确等同论文实验。
- 监控 reward、cost、lambda、KL、actor/critic loss、长度、拒答率、重复率、吞吐与峰值显存；检查保存/恢复。

### P3：正式复现

- 全类别 RM/CM → 全类别 Safe RLHF-V；添加 helpfulness-DPO、safety-DPO、PPO baseline，统一数据/训练预算。
- 做至少多个 seed，并报告均值、置信区间及分类别表现；补齐五个评测集。
- 再考虑 Qwen2-VL / LLaVA-NeXT，以及固定惩罚 vs 动态 lambda、lambda bound、cost severity 等消融。

## 资源建议

现场检查可用节点有 **8 × NVIDIA H20（每卡约 95.6 GiB，标称 96GB），GPU 间 NVLink**，系统内存约 1.9 TiB。适合先单节点 8 卡、bf16、ZeRO-3、gradient checkpointing。策略、reference、RM、CM、reward critic、cost critic 共六个模型实例，其中三个训练实例，容量需实测，不预报虚假的训练耗时。

建议将完整节点预留后再启动，不把“显存还有余量”当作可与其他任务混跑的依据。具体 SSH 地址、密钥位置、磁盘状态与任务占用仅保留在本地私有报告，不发布到公开 fork。

## 分支约定

- `main`：保留上游。
- `reproduction/bootstrap`：复现路线、审计。
- 下一步建议 `reproduction/fix-entrypoints` → `reproduction/environment` → `reproduction/smoke`。
- 所有算法改动在可信 baseline 后实施，避免把工程修复和方法改进混在一起。
