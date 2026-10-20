## Reproduction workspace

This fork adds reproduction tooling. Start with [当前主线：目标、实验计划与推进情况](docs/PROJECT_SUMMARY_ZH.md) and [最新进展（2026-10-20）](docs/PROGRESS_20261020_ZH.md). Historical results: [旧阶段汇报](docs/PROGRESS_REPORT_ZH.md), [复现进展与 RM/CM 状态](docs/REPRODUCTION_ZH.md), [推理实验结论](docs/RESULTS_LLAVA_ZH.md), and [RM/CM 真实训练结果与资源](docs/RESULTS_RM_CM_ZH.md). Technical setup, implementation details, and regression checks are kept separately in [开发记录](docs/development/README.md).

Historical evaluation and engineering results: [旧四策略 Luna PK](docs/LUNA_AUTHOR_PK64_20261010_ZH.md) · [Active Safety v2.1 四轮工程验收](docs/ACTIVE_SAFETY_V2_N32_ZH.md) · [RM/CM sigmoid 指标含义](docs/ACTIVE_SAFETY_METRICS_ZH.md) · [复现与自有算法结果分析](docs/RESULTS_ANALYSIS_20261010_ZH.md). Engineering validation is not a claim of safety improvement or human-risk certification.

**Completed historical LLaVA scorer reproduction:** [作者规模 RM/CM 训练与最终验收](docs/AUTHOR_SCALE_RM_CM_ZH.md) — 29,808 train / 590 evaluation pairs, 3 epochs and 5,589 updates each; full independent reload difference 0. Final RM/CM ranking: 81.86% / 79.49%, CM sign: 86.19%. This is not the new Qwen scorer training and does not certify proxy quality on Qwen responses.

**Latest mainline progress (2026-10-20):** [原生 Qwen 评分器 → 四卡 Joint-PPO 5K](docs/PROGRESS_20261020_ZH.md). 当前先用作者已有公开偏好数据训练 **Qwen3.5-4B 原生 RM／CM**，再接回 **四卡 LoRA Joint-PPO＋PTX／双 critic、5K唯一prompt**。旧代理未通过192条Qwen校准后的质量筛查，因此策略继续暂停。评分器 **5,000训练对＋512内部验证对的CPU准备与全部缓存读回已完成**；正式新RM、CM和本轮策略训练更新均为 **0**，完整GPU保存／尾批／恢复验收仍待完成。当前执行范围为 **仅CPU收尾后停止，不自动启动GPU训练或等待队列**。

研究目标仍是检验有限反馈预算下 Joint-Gradient 相对预算匹配对照的安全性—帮助性收益，**尚无本轮策略效果结论**。评分器使用已有标签，不占625条策略纠正额度；192条既有校准消耗保留记账。Only sanitized documentation is published; private implementation, data, ledgers, logs, weights, credentials and server information are excluded.

**Previous mainline snapshot:** [October 19: 7B共同基线与严格几何验证](docs/PROGRESS_20261019_ZH.md). Those completed no-update checks remain valid historical evidence, not Qwen5K training or proof of method benefit. The separate eight-GPU7B route is deferred. Earlier snapshots: [October 16](docs/PROGRESS_20261016_ZH.md) · [October 17](docs/PROGRESS_20261017_ZH.md).

**Historical progress (2026-10-12):** [资产、恢复与清理状态](docs/PROGRESS_20261012_ZH.md). The old eight-GPU formal PPO run stopped at update 54 without a recoverable formal checkpoint. Six-GPU immediate full restore passed in v4, but continuation parity failed; v5 passed two-fresh-run repeatability only. Full resume and the formal replacement remain unaccepted or unstarted. Authorized cleanup removed old pilot/active policy weights and failed resume snapshots; cached answers, labels and outside-target evidence remain.

**Historical handoff:** [2026-10-11 14:36 项目交接快照](docs/HANDOFF_20261011_ZH.md) — its running-process and artifact-availability descriptions are superseded by the October 12 and October 13 progress reports.

## Introduction

This project is built on top of the [align-anything](https://github.com/PKU-Alignment/align-anything) framework. We introduce new features through the Safe RLHF-V method, enhancing the safety and performance of RLHF multi-modal training.

## Dataset

We use the [BeaverTails-V](https://huggingface.co/datasets/saferlhf-v/BeaverTails-V) dataset for training, a multimodal dataset covering nine primary safety domains, designed to help visual language models detect safety risks and content violations effectively.

## Quick Start

### Easy Installation

```bash
# clone the repository
git clone git@github.com:saferlhf-v/saferlhf-v.git
cd saferlhf-v

# create virtual env
conda create -n saferlhf-v python==3.11
conda activate saferlhf-v
```

- **`[Optional]`** We recommend installing [CUDA](https://anaconda.org/nvidia/cuda) in the conda environment and set the environment variable.

```bash
# We tested on the H800 computing cluster, and this version of CUDA works well.
# You can adjust this version according to the actual situation of the computing cluster.

conda install nvidia/label/cuda-12.2.0::cuda
export CUDA_HOME=$CONDA_PREFIX
```

> If your CUDA installed in a different location, such as `/usr/local/cuda/bin/nvcc`, you can set the environment variables as follows:

```bash
export CUDA_HOME="/usr/local/cuda"
```

Finally, install `saferlhf-v` by:

```bash
# We prepare quick installation for training, you can use the following command:
pip install -e .
```


### Training

We provide some scripts for quick start, you can find them in the `./scripts` directory. These scripts would automatically download the model and dataset, and run the training or evaluation.

For example, `scripts/safe_rlhf_v.sh` is the script for Safe RLHF-V training, you can run it by:

```bash
cd scripts
bash safe_rlhf_v.sh
```


## Wandb Logger

We support `wandb` logging. By default, it is set to offline. If you need to view wandb logs online, you can specify the environment variables of `WANDB_API_KEY` before starting the training:

```bash
export WANDB_API_KEY="..."  # your W&B API key here
```


## Report Issues

If you have any questions in the process of using saferlhf-v, don't hesitate to ask your questions on [the GitHub issue page](https://github.com/saferlhf-v/saferlhf-v/issues/new/choose), we will reply to you in 2-3 working days.

# License

saferlhf-v is released under Apache License 2.0.
