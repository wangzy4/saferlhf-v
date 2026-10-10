## Reproduction workspace

This fork adds reproduction tooling. See [阶段汇报：主线、进展与完成情况](docs/PROGRESS_REPORT_ZH.md), [项目目标、方法主线与当前进度总结](docs/PROJECT_SUMMARY_ZH.md), [复现进展与 RM/CM 状态](docs/REPRODUCTION_ZH.md), [推理实验结论](docs/RESULTS_LLAVA_ZH.md), and [RM/CM 真实训练结果与资源](docs/RESULTS_RM_CM_ZH.md). Technical setup, implementation details, and regression checks are kept separately in [开发记录](docs/development/README.md).

Latest verified results: [旧四策略 Luna PK](docs/LUNA_AUTHOR_PK64_20261010_ZH.md) · [Active Safety v2.1 四轮工程验收](docs/ACTIVE_SAFETY_V2_N32_ZH.md) · [RM/CM sigmoid 指标含义](docs/ACTIVE_SAFETY_METRICS_ZH.md) · [复现与自有算法结果分析](docs/RESULTS_ANALYSIS_20261010_ZH.md). Engineering validation is not a claim of safety improvement or human-risk certification.

In progress: [作者规模 RM/CM 训练](docs/AUTHOR_SCALE_RM_CM_ZH.md) — 29,808 train / 590 evaluation pairs, 3 epochs each. CM has begun real full-parameter training; RM is queued. This does not retroactively change the previous policies, scorers or PK results.

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
