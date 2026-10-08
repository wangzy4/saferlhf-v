# RM / CM 真实试训：方法与资源规划

本页记录工程细节；读者结论见 [RM/CM 结果](../RESULTS_RM_CM_ZH.md)。LoRA 使用独立 PyTorch/PEFT 路径，全参数使用 native score/loss 的独立 DeepSpeed ZeRO-2 runner；不代表原始 trainer 入口已全部通过。

## 第一轮固定配置

- 基座：`llava-hf/llava-1.5-7b-hf`，revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`；三个 safetensors 逐文件校验固定 SHA256。
- 数据：`saferlhf-v/BeaverTails-V`，revision `ee19041205c720c0faea575de563de8a6a8f9094`，只用源 `train.parquet` 构造训练/内部验证。下载校验源 LFS SHA256，不执行 dataset 远端代码。
- 独立环境：从已验证推理环境克隆，增加 `peft==0.14.0`；Python 3.11.17、torch 2.5.1+cu124、Transformers 4.48.3。`pip check` 通过，未修改 inference 环境。
- 原生项目类 `AccustomedLlavaRewardModel`，BF16 backbone，SDPA，训练参数 FP32，autocast BF16。
- `r=8 / alpha=16 / dropout=0.05`，仅语言模型 `q_proj/v_proj` 注入 LoRA；`score_head/multi_modal_projector` 为 `modules_to_save`。视觉编码器冻结。可训练参数 25,178,112；PEFT 模型统计总参数 7,088,609,280（含保存模块副本）。
- 每 microbatch 一对回答（两个完整序列），梯度累积 4，有效 batch 4 对；2 epoch，64 optimizer steps；seed 42；最长 2,048 token（包括 576 个视觉 token），所选样本实际最长 1,835。禁止静默截断回答。
- AdamW：lr `3e-5`、betas `[0.9, 0.95]`、weight decay **0.01**，cosine，3% warmup，gradient norm clip 1.0；语言模型 non-reentrant gradient checkpointing。
- 使用共享基座 processor 的官方 chat template，完整问答＋EOS，以最后有效 token 的 hidden state 给出 scalar score。
- 独立日志为本地 JSON，不上传 W&B，不使用 judge。

### 和上游默认配置的差异

| 项目 | 第一轮轻量验证 | 公开 RM/CM YAML 默认 |
| --- | --- | --- |
| 参数更新 | 语言 q/v LoRA＋投影＋评分头 | 语言模型＋投影＋评分头，冻结视觉 |
| epoch | 2 | 3 |
| context | 2,048 | 4,096 |
| RM scheduler | cosine | constant_with_warmup |
| CM scheduler | cosine | cosine |
| weight decay | 0.01 | 0.0 |
| batch | 每卡 1 对，累积 4 | 每卡 4 对，累积 1；实际作者卡数未知 |
| execution | 单 GPU 原生 AdamW/PEFT | DeepSpeed |
| 数据 | 五类 128/32，train 内划分 | 实际作者数据清单/划分未取得 |

因此保留这轮作为工程与小规模能力诊断，不标成论文复现。论文 Appendix A.1 给出了初始化及损失，但没有完整覆盖这些实际运行参数；公开默认配置也不能直接当成作者实验的全部配置。

## 数据准备与隔离

`scripts/prepare_rm_cm_smoke.py`：

1. 下载固定 revision 的选定类别 `train.parquet`，校验 SHA256。
2. 从全部 20 个 evaluation parquet 读取图像，仅用于重复排除，不据其评分选训练超参数。
3. 通过 RGB 图像尺寸＋像素 SHA256 分组。590 evaluation 行对应 589 个唯一图像。
4. 过滤相同回答、非法偏好 ID；按类别和 seed 打乱，先取内部验证，再取训练，所有所选图像互不重复。
5. processor 展开视觉 token 后检查长度；超长候选排除、不截断；所选样本图片 token 数必须为 576。
6. 产物为私有 `train.parquet/validation.parquet/manifest.json/READY`。公开只提交聚合清单和哈希，不提交图片或回答。

扩大规模时 `--all-categories` 使用全部 20 类，`--exclude-dataset` 排除上一轮两个 split 用过的图像。新增检查 rating 与安全标签符号一致，冲突直接失败、不修改标签。检查只覆盖精确解码图像重复，不是近似图像/语义泄漏认证。

## 损失与评分

`safe_rlhf_v/utils/preference_training.py` 与实际上游 RM/CM `loss()` 做数值及梯度等价测试：

- RM：`-log sigmoid(R_helpful - R_other)`＋分数平方均值正则（0.001）。
- CM：`-log sigmoid(C_other - C_safer)`＋两个回答各自的 `-log sigmoid(-harmless_rate * C)`＋相同正则；`scale=1`。
- `harmless_rate=0` 的绝对项保持常数、梯度为 0，不改写标签。更高成本代表更有害。
- 排序准确率使用严格 `high > low`，平分判为不正确。与策略 mean-token-logp 指标的 tie=0.5 口径不同。
- CM 阈值诊断中 `cost>0` 判为有害；记录多数类基线及平衡准确率。后续增加逐类别统计、confusion matrix 和 response-level AUC（相同分数计 0.5），用于区分排序学习与阈值校准问题；不是生成安全评测。

## 真实初始化与产物检查

HF 的基座到 score wrapper 前缀映射可能没有报告外部 `score_head.weight` missing key。第一轮 v1 因预设其必须出现而停止，**尚未开始训练**；文件保留，标记失败。v2 独立检查 backbone 完整 key inventory、无 meta tensor、四个关键 tensor 与固定基座权重完全一致，并显式初始化新 Linear score head，再训练。

保存产物包含 LoRA、投影层和评分头及 processor，约 97 MiB/模型；加载必须使用固定基座和本项目 score class。重新加载全部基座＋adapter 后，两对内部验证抽查的 score 最大绝对差为 0。没有检验 optimizer/RNG 的断点续训，也没有检验合并后完整 checkpoint 或 RL 加载。

## 运行方式

所有大文件、环境、缓存和日志放在用户指定的数据盘根目录。启动前检查资源，`CUDA_VISIBLE_DEVICES` 必须是一个明确物理 GPU 编号；脚本在加载前重新检查显存，发现占用即拒绝。多用户机器没有原子 GPU 预留，外部任务可能在检查后启动，不能把快照当预留。

```bash
export ROOT=/path/on/data/saferlhf-v
export HF_HOME="$ROOT/hf-cache"
export HF_DATASETS_CACHE="$ROOT/hf-cache/datasets"
export TMPDIR="$ROOT/tmp"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
PY="$ROOT/envs/rm-cm-smoke/bin/python"

CUDA_VISIBLE_DEVICES="" "$PY" scripts/prepare_rm_cm_smoke.py --root "$ROOT"
CUDA_VISIBLE_DEVICES=<one-free-gpu> "$PY" scripts/train_preference_smoke.py \
  --root "$ROOT" --kind rm --run-name rm-cm-128-32-lora-r8-v2
# 另一张空闲 GPU 可独立运行 --kind cm；数据和起始配置相同。

CUDA_VISIBLE_DEVICES="" "$PY" scripts/prepare_rm_cm_smoke.py --root "$ROOT" \
  --name rm-cm-20cat-1024-256 --train-size 1024 --validation-size 256 \
  --all-categories --exclude-dataset rm-cm-smoke-128-32
CUDA_VISIBLE_DEVICES=<one-free-gpu> "$PY" scripts/train_preference_smoke.py \
  --root "$ROOT" --kind cm --run-name rm-cm-20cat-1024-256-lora-r8-v1 \
  --dataset-name rm-cm-20cat-1024-256
```

不同配置必须用新 run 名；脚本不覆盖既有 run。新阶段保持第一轮优化配置，从基座重新训练，不续接第一轮 adapter。环境复用、评测统计增强不改变训练算法。

## 资源与正式复现规划

### 已测：轻量版本

- 每模型 1×H20，峰值 allocated 15.36 GiB / reserved 16.68 GiB；任务总用时约 6 分钟，纯训练约 3.85 分钟，两模型可并行。
- 五类源 train parquet 约 0.74 GiB；全部 20 类约 **3.53 GiB**。第一轮选定数据约 21 MiB，两个 adapter 合计约 194 MiB。环境单独 `du` 约 5.8 GiB，克隆可能共享硬链接，不能据此计算净新增物理空间。
- 第二轮 1,024/256 已实测：每模型纯训练约 31 分钟、加载/评估/保存/重载总计约 42 分钟；峰值 allocated/reserved 15.47/16.68 GiB。

### 未测：论文/公开配置方向

论文 Table 4 的 5K / 15K / 30K 评分质量是规模参考，不直接与小内部验证比较。建议先准备 5K 量级训练、覆盖全部类别、固定 train 内验证，再逐步扩大。

全参数路线按“冻结视觉、训练语言模型/投影/评分头”估算：

- 约 6.7B 可训练参数的常规混合精度 Adam 按 **16 bytes/参数**（权重、梯度、FP32 主权重与动量）估计，静态状态约 **100 GiB**，还不含激活、视觉模型、通信和临时 buffer。具体 Adam/参数精度会改变此值。
- 因此不能把单卡 16 GiB 的 LoRA 实测套用于全参数训练，也不建议未经试跑承诺单张 96GB H20 全参数 Adam 可容纳。
- 初步规划单模型 **2–4 张 H20＋ZeRO**，优先用 4 卡做小规模真实全参数预检；先实测峰值/速度再确定正式卡数。2 卡能否容纳取决于 batch、context、ZeRO stage、激活及通信开销，尚未认证。
- 完整 BF16 checkpoint 每个约十余 GiB，optimizer 状态另占数十 GiB；预留每模型约 100–200 GiB 运行空间是规划预算，不是产物实测。
- RM/CM 全参数试跑建议先串行，避免一次占满八张卡；必须当时重新检查资源。

完整 trainer 已能在独立环境导入；原生 score class 已通过随机 tiny 的 4 卡 NCCL/ZeRO-2/FusedAdam 更新及恢复。v1 真实 7B 已完成优化更新并实测资源，但发现非持久 RoPE buffer 的保存/重载合同缺口，不能把 tiny 或单对抽查当成完整 7B 加载认证，也不是原始入口全部认证。

### 第二轮：已完成

- `rm-cm-20cat-1024-256-lora-r8-v1`，启动代码 `9abc235299a69074c8310ab6fad42a6de7fab951`。
- 数据已准备：全部 20 类、1,024 条训练＋256 条内部验证，seed 42，最长所选 1,953 token。
- 扫描全部源 train，精确图像排除统计：与 evaluation 重复 2 行、与第一轮 160 张已用图像重复 176 行；取样时另跳过 3 条重复图像。新的两个 split 互不重叠，也不与第一轮或 evaluation 精确图像重叠。
- RM/CM 各单 GPU，从固定基座重新训练，预定 2 epochs 全部完成，不选择第 1 epoch 的 RM 更高值。最终内部验证：RM 201/256（78.515625%）；CM 205/256（80.078125%），零阈值安全标签 412/512（80.46875%）、balanced 79.9089%、AUC 0.882765、majority 59.375%。两模型 fresh-base adapter 重载分数差异均为 0。新旧验证集不同，结果差值不是严格规模消融，未在论文 evaluation 上调参。

### 独立全参数训练环境：前置检查进展

已建立独立 `rm-cm-full` 环境，不改 inference/smoke/其他项目环境：

- 保持 torch 2.5.1+cu124 / Transformers 4.48.3；增加 DeepSpeed 0.16.2、diffusers 0.32.2、torchaudio 2.5.1+cu124、librosa 0.10.2.post1、tensorboard 2.18.0、wandb 0.19.1、scipy 1.15.1、rich 13.9.4、OpenCV headless 4.10.0.84 等当前导入所需依赖；`pip check` 通过。
- 原始 `text_image_to_text.rm.RMTrainer / cm.CMTrainer` **真实导入通过**，43 项 CPU 测试无跳过通过。没有启用外部日志上传。
- 安装环境内 CUDA nvcc 12.4.131、cudart-dev / cccl 12.4.127；不安装系统驱动、不改系统 toolkit。
- 隐藏全部 CUDA 设备，使用 DeepSpeed 原始 FusedAdam 源码和编译 flags 针对 H20 的 `sm_90 / compute_90` 交叉编译，扩展构建与导入通过，缓存全部在数据盘。
- DeepSpeed 0.16.2 原生 JIT 在无可见 GPU 时忽略 cross-compile architecture 参数、设备列表为空导致失败；交叉编译改用 torch 扩展 loader。补充 PyTorch CUDA wheels 的 include 路径以提供 cuSPARSE/cuBLAS 等 headers，未修改安装包源码。

**环境导入/交叉编译本身不是完整训练成功。** tiny 4 卡验证实际 GPU 更新、NCCL、ZeRO-2 分片与恢复；真实 7B v1 实测见结果页，重载合同正在修正。GPU JIT 通过 `CPATH` 带齐 PyTorch CUDA wheels 的 `nvidia/*/include` 搜索路径，没有修改包源码。

### 小规模全参数 runner 与多卡预检

新增 `scripts/train_preference_full.py`、`utils/preference_distributed.py`，使用同一 native score class / 上游等价 loss，不使用 LoRA，不是原始框架入口的完整认证。

- 已完成 `ds-score-tiny-4gpu-v2`：4 个物理 GPU、NCCL、ZeRO-2、BF16、FusedAdam、4 次同步更新；随机小架构的检查不是 7B 学习结果。
- 保存全模型 safetensors 与 DeepSpeed 模型/optimizer/scheduler checkpoint；故意扰动 head 和 optimizer moments 后恢复，检查 head 分数及 moment probes 完全一致；恢复后再做一次真实更新且不覆盖最终保存产物。
- 独立 fresh model 加载无 missing/unexpected/mismatched/error 或 meta 参数，抽查分数差异 0；checkpoint 恢复分数差异 0。该检查不认证跨 world size、RNG/数据游标恢复或 ZeRO-3。
- 完整语言骨干＋投影层＋评分头训练，冻结视觉及与标量损失断开的词表输出头；weight decay 0 时后者本来无更新。不认证其他架构或 tied vocabulary head。
- `num_logits_to_keep=1` 只减少不用于 loss 的词表 logits，保留完整 hidden states；CPU tiny 测试验证 end scores 完全一致、语言与 projector/head 梯度非零。
- 全参数数据新建 20 类、256/64，排除两轮 LoRA 用过的全部 1,440 张图像及 evaluation 精确图像重复；与此前训练/验证不混用。
- 计划 3 epochs、每卡 4 对、全局 16 对、48 updates；FusedAdam lr3e-5 / betas0.9,0.95 / wd0，RM constant_with_warmup、CM cosine，warmup `int(48*0.03)=1`。使用 ZeRO-2 而非上游示例 ZeRO-3；原作者实际配置仍未知。
- v1 RM 实测：训练 286.69 秒、原检查全程 540.52 秒，各 rank 最高 allocated/reserved 57.91/64.51 GiB；完整模型 14,127,541,846 bytes、DS checkpoint 94,540,146,055 bytes，约 101 GiB/模型。CM 完成 48 updates 后 fresh 模型检查失败，无 DONE/summary。修正轮扩展验证的成本另测，不将 v1 时间当成新协议实测。

运行时额外设置环境内 `CUDA_HOME`、数据盘 `TRITON_CACHE_DIR/TORCH_EXTENSIONS_DIR`、`CPATH` 指向 CUDA wheels include；公开 [环境版本](../../environments/rm-cm-full-pip-freeze.txt)不含机器路径。启动前逐卡检查，空闲快照不构成预留；checkpoint 只重载本轮自产数据，不使用作者远端 pickle。

```bash
# GPU_LIST 必须是当时空闲、显式选定的四个物理 GPU 编号。
CUDA_VISIBLE_DEVICES="$GPU_LIST" "$ROOT/envs/rm-cm-full/bin/torchrun" \
  --standalone --nproc_per_node=4 scripts/train_preference_full.py \
  --root "$ROOT" --run-name <new-run-name> \
  --dataset-name rm-cm-full-20cat-256-64 --kind rm
# RM 成功且重新检查资源后，另启 --kind cm；不并行占八卡。
```

## 非持久 RoPE buffer 的重载修正

v1 训练源 `08d3537aa488be6959898695fb2b7dbd1288fdbc`。CM 训练 48 updates 完成，但独立 HF 重载未通过原 `atol=rtol=1e-3` 检查。RM 原先只检查一对并通过，后续也不能据此认定完整协议已认证。

1. CPU 对比 HF safetensors 与本轮自产 DS `module`：RM/CM 所有键、dtype、shape、tensor 完全一致，不是权重写坏。
2. 同一 CM 权重、processor、输入、SDPA 及 BF16 参数，用保存配置 fresh load 和原初始化架构重建，首个配对的最大分数差为 0.0078125（逐样本分数只留私有记录）。
3. 唯一不同的 buffer 为 `model.language_model.model.rotary_emb.inv_freq`：fresh FP32，DeepSpeed/整体 `.bfloat16()` 后 BF16；频率最大绝对差 0.001223146915435791。它 `persistent=False`，不出现在 state_dict，权重完全一致检查无法发现。
4. 仅对齐这一 buffer，第一对分数完全一致。64 对 canonical/旧精度执行的最大分数差 0.017578125；排序计数和零阈值计数本轮恰好相同，不代表执行差异可忽略。
5. 修正 `restore_llama_rope_fp32()` 在 `deepspeed.initialize()` 后，用 Transformers 原生 `rope_init_fn` 在 CPU 重建频率并以 FP32 搬到实际设备，同时同步 `original_inv_freq/attention_scaling`。不能仅 `.float()`，因为 BF16 舍入已经丢失信息。仅认证默认 Llama RoPE，其他 RoPE scheme 拒绝而非静默近似。
6. 保持原定数据与优化设置，新 run 从固定基座重训；保留 v1，不放宽容差。扩展 fresh 检查到整个内部 validation，同 rank-strided batch 组成，额外检查 preference 排序和 CM 零阈值判定一致。

修正后的 `ds-score-tiny-4gpu-v4-rope-fp32` 已通过：随机 tiny 架构，1,024 train/256 validation、1 epoch、256 updates，4 卡 ZeRO-2/FusedAdam；整个 256 对 fresh 分数最大差 **0**，optimizer moment 恢复和真实 post-restore update 通过。不是 7B 效果证据。CPU 回归额外覆盖 BF16 频率舍入、非持久 buffer、原生重建和 HF 完整保存重载。

## HF 产物导出

`scripts/export_score_artifacts.py` 仅接受 DONE/summary/reload 通过的 run；全参数额外要求 FP32 RoPE 与整个 validation 重载认证，旧单对协议不允许导出。只白名单导出 safetensors、processor/tokenizer、净化配置、聚合结果、许可文件和 SHA256 清单；绝对 base 路径改为固定 HF ID/revision，未知私有内容直接失败，不静默删除。

上传显式 `--upload`，默认私有；只有显式 `--public-on-private-quota` 且确认私有存储配额错误才切公共，授权或网络错误不能触发公开。上传后按 immutable HF commit 核对远端权重 LFS SHA256，并下载 metadata 验证 SHA256，保存 receipt。只归档推理权重，不把 optimizer/RNG 重启合同冒称云端备份。第二轮两个 adapter 已私有归档；全参数 v1 暂不作为有效产物上传。HF OAuth 凭证及机器侧授权实现仅留私有数据缓存，不提交代码库。

## 检查记录

新增 RoPE 与产物 privacy/public-fallback 回归后，**50 项测试**在独立 full 环境、隐藏 CUDA 的 CPU 全部通过，无跳过。此前46项覆盖全参数冻结/梯度、词表 logits 裁减评分等价、全局 batch 合约；此前43项包括实际 trainer loss 数值/梯度等价、标签方向、0 rating、真实 tiny score LLaVA 的 LoRA 更新与 adapter 保存重载、cost diagnostics、以及训练 mask/critic/rollout 合约。真实 7B 训练和测试证据分别报告。

没有调用付费 API，没有上传逐样本数据，没有恢复暂停的 judge 下载。第一轮权威聚合见 [结果 JSON](../results/rm-cm-128-32-lora-r8.json)。
