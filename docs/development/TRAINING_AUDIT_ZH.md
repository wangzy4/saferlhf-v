# RM / CM / Safe RLHF-V 训练链路审计

## 范围与结论边界

本轮针对 LLaVA-1.5 路径核查偏好标签、成本方向、终点计分、rollout 掩码、critic 配置和拉格朗日乘子更新，并修复可复现的工程错误。

**没有训练新的 RM/CM，没有运行完整 DeepSpeed 或 Safe RLHF-V 训练，也没有修改已发布策略权重。** 本轮不能被解释为策略安全性提升；此前有效推理结果保持不变，见 [RESULTS_LLAVA_ZH.md](../RESULTS_LLAVA_ZH.md)。不安排新增本地 judge 模型，也未调用付费评判服务。

## 1. CM 标签方向正确，不应翻转

相关代码：

- `safe_rlhf_v/configs/format_dataset.py`：`RM_V`、`CM_V`。
- `safe_rlhf_v/datasets/text_image_to_text/preference.py`：CM 数据与 collator。
- `safe_rlhf_v/trainers/text_to_text/cm.py`：多模态 CM 继承的损失。

RM 的 `better` 是 **更有帮助**的响应。CM 则有意把 `safer_response_id` 放在 `worse` 一侧：这里 `better` 表示 **应该具有更高成本**，并非更安全。

令 `C_high` 对应相对更不安全响应，`C_low` 对应相对更安全响应，pairwise 项为：

```text
-log sigmoid(C_high - C_low)
```

梯度下降提高 `C_high`、降低 `C_low`。绝对分类项使用 `s = -harmless_rate`：

```text
-log sigmoid(s * C)
```

因此有害响应的负 harmless rating 变成正 `s`，推动 cost 为正；安全响应的正 rating 变成负 `s`，推动 cost 为负。与 actor 的 `reward_advantage - lambda * cost_advantage` 方向一致。

**保留原 CM 顺序和取负逻辑。** `is_better_safe` / `is_worse_safe` 的名字容易误导：它们实际是有符号的分级标签，不是布尔变量。其数值也不意味着 CM 在直接回归相同数值的成本。

### 0 分标签和 evaluation 标注检查

固定 BeaverTails-V revision `ee19041205c720c0faea575de563de8a6a8f9094`，只检查已下载的 20 类 **590 条 evaluation**：

| 检查 | 结果 |
| --- | ---: |
| 响应级标签数 | 1,180 |
| harmless rating 为 0 | 154 |
| `(rating >= 0)` 与 `is_response_safe == yes` 一致 | 1,180 / 1,180 |
| helpful 偏好选中响应的 rating 更高 / 相等 / 更低 | 520 / 70 / 0 |
| safer 偏好选中响应的 rating 更高 / 相等 / 更低 | 399 / 189 / 2 |

两个 safer 偏好与分级评分的次序不一致样本，**均未跨越安全/不安全标签边界**。不能据此直接改写偏好 ID：pairwise 偏好和分级评分是不同标注，需进一步检查。

`rating = 0` 时绝对项恒为 `log(2)`，对 cost 的梯度为零；pairwise 项仍然有效。由于 0 分对应的响应在该 evaluation 中标为安全，绝对项并不单独把这些响应推入负成本区域。这是需要训练后校准/消融的语义注意点，不是本轮擅自将 0 改成 -1 的理由。

这些检查是**数据标签审计，不是模型准确率**。没有下载或检查 train split，尚未检查跨 split 泄漏。

## 2. 修复 LLaVA score head 对右 padding 的错误终点计分

原 `AccustomedLlavaRewardModel.forward` 固定读取 `last_hidden_state[:, -1, :]`，而 RM/CM 加载路径使用右 padding。短响应的 batch 最后位置可能是 padding，而不是真实响应终点。同时，原 `end_index` 是 CPU 浮点型的 `-1`，与输出声明的整型索引不符。

现在：

- 按 attention mask 找到每行最后一个有效位置，支持左右 padding。
- 从同一位置提取 `end_scores` 和 `end_last_hidden_state`。
- 返回与 hidden states 同设备的 `torch.long` 索引。
- 空序列、非二值 mask、mask 与 hidden states 长度不匹配时明确报错。
- 未提供 attention mask 时视为全部位置有效；含 padding 的调用必须显式提供 mask。

新增 `utils/masking.py`。训练 loader 也为原生 LLaVA processor 显式设置 patch size、CLIP CLS token 数和视觉选择策略，并同步 processor tokenizer 的 `model_max_length`，避免未展开图像占位符与 hidden-state 位置不一致。

实际小型随机 LLaVA 的 CPU 回归确认：右 padding batch 的终点索引为 `[6, 8]`，终点分数与各自不含 padding 的单条前向一致，score head 反向梯度有限且非零。**不是 7B checkpoint 或已训练 RM/CM 的效果测试。**

## 3. 修复 rollout 与 RL 更新中的 padding 污染

原 Safe RLHF-V 路径存在几项相关错误：

1. 用 `log_probs != 0` 猜测有效 token。有限精度下合法 logp 可以为 0，不能当作 padding 标记。
2. 对单 token 响应的标量额外补两个零，凭空将长度扩成 3。
3. `rl_step` 用 `ones_like(response_mask)` 覆盖真实 mask，所有右侧 padding 被计入 terminal reward/cost、GAE、损失和长度统计。
4. rollout 保存的是 `reward_batch['input_ids']`；当 scalar RM 重分词时，actor 训练可能拿到错误 token ID。

现在统一根据真实 `response_lens` 构造 mask；单 token 响应保持长度 1；RL 更新保留真实 mask，并检查 logp、critic values 和 mask 的形状一致。KL 项在 padding 位置清零，terminal reward/cost 放在每行最后有效位置。actor 输入保留 actor 的 ID 与 attention mask。

回归覆盖：合法 0-logp token、1-token 响应、长短混合 batch、padded/unpadded GAE 一致、padding 梯度为零，以及终点 reward/cost 的放置。

## 4. 修复 cost critic 配置与 tokenizer 路由

- 原代码忽略 `cost_critic_model_name_or_path`，总从 CM checkpoint 初始化 critic。现在优先使用显式 critic 路径；未设置时回退到 CM，保持原示例可用。
- cost critic 现在与 reward critic 一样遵守 vision tower / projector / language model 的冻结配置。
- 原代码在 actor 与 **cost** tokenizer 相同时却替换 **reward** tokenizer，随后又无条件覆盖 cost tokenizer。现在分别比较 scalar RM、CM tokenizer，仅在实际匹配时复用 actor tokenizer。
- reward critic 和 cost critic 都必须与 actor tokenizer 匹配，因为 value 是 tokenwise 的；不匹配立即报错，不能靠换一个 tokenizer 对象掩盖模型词表差异。

保留 scalar 模型已有的重分词路径不代表已经验证跨架构图像 processor 兼容性；本轮并未认证任意 RM/CM/actor 组合。

## 5. lambda 与 KL：方向正确，精确更新形式需要区分

当前组合优势：

```text
A = (A_reward - lambda * A_cost) / (1 + lambda)
```

reward 使用负 KL 惩罚，cost 使用正 KL 项。上述组合中两者合成相同强度的负 KL 惩罚，而非重复放大 KL。不能只看到两个 KL 项就删掉其中一个。

当前 dual loss：

```text
lambda_loss = -(mean_episode_cost - threshold) * exp(log_lambda)
```

成本超过阈值时提高 lambda，低于阈值时降低 lambda；达到上界后对 `log_lambda` 截断。默认配置为 threshold `-0.5`、lambda_init `10`、lambda_max `20`、lambda_lr `0.1`、delay `1`、cost window `128`。

需要明确：实现是在 **log(lambda) 参数上做 SGD**，未触及截断时等价于：

```text
lambda_new = lambda * exp(lr * lambda * (mean_episode_cost - threshold))
```

这与论文式 (8) 直接对 lambda 做加性投影更新的**精确步长并不相同**，虽方向和上界意图一致。本轮保留原优化方式，不将其未经实验就改成另一种算法。CPU 测试验证了上升、下降、上界和 delay，但没有验证真实多卡通信或训练稳定性。

## 6. 已执行验证与限制

新增 `tests/test_training_contracts.py` 的 19 个测试，与已有 15 个测试合计：

- 独立推理环境（Python 3.11.17、torch 2.5.1、Transformers 4.48.3），隐藏 CUDA：**34 / 34 通过，无跳过**。
- 不含 PyTorch 的本机基础 Python：24 个通过，10 个 tensor/native-model 测试明确跳过；不把 skip 当 pass。
- 包内 **44 个 Python 文件 AST 检查通过**；compileall 和差异空白检查通过。
- 对本轮修复前提交 `59cc1ed`，只补入测试及测试所需 helper、不替换原模型/训练代码：新增 19 个测试中 **8 个失败**；相同测试在修复后全部通过。

测试从实际源文件 AST 加载方法/类定义，绕过 optional 训练包导入；输出容器、训练引擎和分布式通信使用轻量替身，更新使用 CPU SGD。随机小型 LLaVA 使用真实 Transformers 网络做前向/反向。**这些是工程回归，不等于完整训练包可导入、DeepSpeed 可启动、checkpoint 可保存恢复、RM/CM 质量合格或完整 RL 复现成功。**

下一步优先准备独立训练环境，核查缺失初始化、optimizer/toolkit、保存恢复等路径，再做小规模真实 RM/CM 训练 smoke test。LLaVA 缓存生成需继续使用已验证的 Transformers 4.48.3，并保留单条/批量回归；不能假定所有未来版本兼容。完整训练、train/evaluation 去重和其他模型架构仍待验证。
