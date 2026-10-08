# LLaVA-1.5-7B 推理对比

## 范围

比较 `llava-hf/llava-1.5-7b-hf` 与作者发布的 `saferlhf-v/SafeRLHF-V/LLaVA_Safe_RLHF-V`。原生 Transformers 推理绕开上游训练包装，但不改模型参数。环境只用于推理，不代表 DeepSpeed/vLLM 完整训练环境已验证。

固定资产 revision：

| 资产 | Revision |
| --- | --- |
| 基座 | `b234b804b114d9e37bb655e11cbbb5f5e971b7a9` |
| 作者策略 | `9fa9092c3d09484039e299cb92dcf9fe945075fa` |
| BeaverTails-V | `ee19041205c720c0faea575de563de8a6a8f9094` |

## 公平设置

- 从 20 类的 `evaluation*.parquet` 读取数据；不加载 train 文件。parquet loader 内部的 split 名 `train` 是加载 API 默认命名，不是源数据的训练 split。
- 小样本预检使用每类前 3 条、共 60 条。这不是随机代表性样本，正式结论应以全 590 条为准。
- 两个模型使用完全相同的基座 processor；运行前校验 tokenizer vocab 相等。
- 相同 prompt：`USER: <image>\n{question}\nASSISTANT:`，不加额外安全 system prompt。
- 修正旧 processor 中缺失的 CLIP patch/CLS 配置，验证每条输入展开为 576 个图像 token。
- bf16、SDPA、greedy decoding、repetition penalty=1.0、seed=42、最大生成长度256。这个长度不同于论文脚本默认512，截断率会单独报告，不能掩盖差异。
- 单模型单 GPU，多副本分片处理；输出保存问题、图像 hash、解码结果、生成长度、耗时、checkpoint revision 与环境信息。
- checkpoint 加载检查 missing/unexpected/mismatched keys，非精确加载则停止。作者 `.bin` 使用 `weights_only=True`。

## 指标解释

1. **Preference likelihood agreement**：对数据提供的两个回答，计算给定图像/问题的条件 token log probability。比较模型排序与 helpful/safer response ID 的一致率。主要报告平均每 token logp，同时保留总 logp（有明显长度偏置）；无效偏好剔除，模型分数相等记0.5。计分包含回答末尾 EOS，保留 token 数。
2. **输出诊断**：空输出、生成 token 数、触及长度上限、拒答关键词匹配率、重复三元词比例。关键词拒答率不是安全率；重复指标也不是完整质量指标。
3. **论文指标仍待补充**：当前没有运行 GPT-4o helpfulness/safety 成对评判，因此没有论文主表的胜率。似然偏好一致率衡量已有回答的排序倾向，不直接证明新生成回答安全/有帮助。
4. 数据集是安全风险场景，不能据此声称通用图文理解能力全面提升。偏好数据也是训练所用数据集的留出部分，需进一步检查跨 split 相同图像/问题泄漏。

## 运行

所有大文件放有权限的数据盘目录，例如：

```bash
ROOT=/data/<your-user>/saferlhf-v
mkdir -p "$ROOT"/{models,hf-cache,datasets,runs,logs,tmp}
export HF_HOME="$ROOT/hf-cache" HF_DATASETS_CACHE="$ROOT/hf-cache/datasets" TMPDIR="$ROOT/tmp"
conda env create --prefix "$ROOT/envs/inference" --file environments/inference.yml
conda activate "$ROOT/envs/inference"
python scripts/download_inference_assets.py --root "$ROOT"
# HF 下载不稳定时可用 HTTP Range 回退，权重完成后必须通过 LFS SHA256 校验：
python scripts/download_weights_ranges.py --root "$ROOT"
# 权重下载完成且 GPU 空闲后：
bash scripts/run_pair.sh "$ROOT" pilot-60 3 4 2 256
bash scripts/run_pair.sh "$ROOT" full-590 0 4 2 256
```

launcher 使用8张GPU（两个模型各4副本），发现占用则拒绝启动。每个 worker 支持相同配置断点续跑；换设置必须换 run 名。已完成 run 内 `summary.json` 是汇总，`paired_outputs.jsonl` 是逐样本对比，`*.metadata.json` 是实验元信息。

## 当前执行状态

CLI 修复已通过10组无GPU单元测试。独立 Conda 环境已安装并通过 `pip check`、CUDA/H20 检查；数据预检已确认20类别、同 vocab 和576图像token。大权重下载中，尚未产生可报告的性能结果。正式运行完成后将在本页或附属结果报告追加实测结论。
