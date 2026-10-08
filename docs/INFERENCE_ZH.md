# LLaVA-1.5-7B 推理对比

## 范围与状态

比较基座 `llava-hf/llava-1.5-7b-hf` 与作者发布的 `saferlhf-v/SafeRLHF-V` 中的 `LLaVA_Safe_RLHF-V` 子目录。原生 Transformers 推理绕开上游训练包装，不改模型参数。**已完成修复后 590 题同题推理和偏好似然计分**，结果见 [实测报告](RESULTS_LLAVA_ZH.md) 和 [逐类别聚合指标](results/llava-590-tf4483.json)。环境只用于推理，不代表 DeepSpeed/vLLM 完整训练环境已验证。

固定资产 revision：

| 资产 | Revision |
| --- | --- |
| 基座 | `b234b804b114d9e37bb655e11cbbb5f5e971b7a9` |
| 作者策略 | `9fa9092c3d09484039e299cb92dcf9fe945075fa` |
| BeaverTails-V | `ee19041205c720c0faea575de563de8a6a8f9094` |

四个大权重文件均通过上游 LFS SHA256 校验。所有模型、数据、缓存、环境及运行输出保存在数据盘，不提交 Git。

## 公平设置

- 20 类的 `evaluation*.parquet` 共590题，不加载 train 文件。parquet loader 内部的 split 名 `train` 是加载 API 命名，不是源数据的训练 split。
- 两个模型共享固定基座的 slow processor，运行前校验 tokenizer vocab 相等。修复缺失的 CLIP patch/CLS 属性，逐条检查576个图像token。
- 主实验使用基座官方 chat template，即 `USER: <image>\n{question} ASSISTANT:`；不加额外安全 system prompt。作者发布 processor 没有 chat template，因此显式共享基座模板，而不依赖缺省 fallback。
- bf16、SDPA、greedy decoding、repetition penalty=1.0、seed=42、max_new_tokens=512。
- 两模型各3副本，每副本单GPU，batch_size=4；每模型处理590题。此设置不是论文 GPT-4o 评测的完整复现。
- 检查 checkpoint missing/unexpected/mismatched keys，非精确加载则停止。作者 `.bin` 使用 `weights_only=True`；本次六个加载检查均为空。
- 输出保存问题、图像 hash、响应、生成长度、偏好标签、条件似然、耗时和环境元信息；汇总前检查两模型的样本与配置一致。

## 关键兼容性修复

初始 Transformers4.47.1 在已展开图像token、左padding、cached batch decoding 的组合下错误重建 attention mask。同一道题单条生成正常，批量生成却循环重复首词。定位到其 legacy decoding 分支后，升级到 **4.48.3**；两模型的同一 padding regression batch 均恢复正常。

**之前4.47产生的三个run只作为问题诊断，不用于生成质量结论。** 非缓存似然计算不经过上述错误分支，但最终报告也全部重新计分，统一使用4.48.3。推理脚本现拒绝旧版本，GPU回归脚本见 `scripts/check_llava_padding.py`。这一结论针对本次推理组合，不能据此认定作者历史训练/评测均受到同一问题影响。

## 指标解释

1. **Preference likelihood agreement**：对数据已有的两个回答，计算给定图像/问题的条件 token log probability，比较排序与 helpful/safer response ID 的一致率。主要报告平均每token logp，也保留总logp（有明显长度偏置）。计分包含EOS；无效标签剔除，相等记0.5。
2. **输出诊断**：空输出、token长度、长度上限、拒答关键词、重复三元词比例及连续重复片段。关键词拒答率不是安全率；这些不是完整质量评判。
3. **计算性能**：同一batch的生成耗时只计一次；按样本分摊后汇总所有副本。时间不含模型加载、图片预处理及偏好计分，不是单请求延迟；tokens/s 是归一到单副本的批处理吞吐，不是多个GPU并行的整体吞吐。
4. **论文指标待补充**：没有调用 GPT-4o judge，因此没有论文 helpfulness/safety pairwise win rate。偏好似然衡量已有回答的排序，不直接证明新生成回答安全/有帮助。
5. 这是安全风险数据集留出集，不能代表通用图文理解能力；仍需检查跨split图像/问题重叠。

## 运行

```bash
ROOT=/data/<your-user>/saferlhf-v
mkdir -p "$ROOT"/{models,hf-cache,datasets,runs,logs,tmp}
export HF_HOME="$ROOT/hf-cache" HF_DATASETS_CACHE="$ROOT/hf-cache/datasets" TMPDIR="$ROOT/tmp"
conda env create --prefix "$ROOT/envs/inference" --file environments/inference.yml
conda activate "$ROOT/envs/inference"
python scripts/download_inference_assets.py --root "$ROOT"
# HF下载不稳定时可用可续传的Range回退，强制校验权重SHA256：
python scripts/download_weights_ranges.py --root "$ROOT"
# 仅在选定GPU空闲时运行；这里的卡号是用法示例，不是资源预留：
CUDA_VISIBLE_DEVICES=0 python scripts/check_llava_padding.py --root "$ROOT"
GPU_IDS=0,1,2,3,4,5 bash scripts/run_pair.sh "$ROOT" full-590-hf-greedy512-tf4483 0 3 4 512 hf
```

launcher 检查所选GPU占用，拒绝混用已有任务；不设GPU_IDS时默认使用前 `2*REPLICAS` 张卡。每worker支持同配置断点续跑，换设置必须换run名。`summary.json` 是汇总，`paired_outputs.jsonl` 是逐样本对比，`*.metadata.json` 是元信息。

## 工程验证

15组无GPU单元测试通过，42个库内Python文件AST解析通过。独立Conda环境通过 `pip check` 和CUDA检查；已执行实际checkpoint的单条/左padding批量生成回归，并完成全量590题两模型生成与计分。完整训练流程、RM/CM训练及论文judge胜率仍未验证。
