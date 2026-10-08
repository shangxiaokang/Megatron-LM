# QwenAir 训练实现方案与验收计划

状态：开发中（2026-10-08）。本文件列出目标、实现顺序与可验证的完成条件；任何阶段只有取得对应运行证据后才可标记完成。

## 1. 冻结输入与目标

- 目标配置：`configs-and-numbers/sglang/agg/accuracy/numbers/provenance/bf16-model-config.json`，SHA-256 `b7d4f14b5891c998e767f92637e1f9c81eca57252f645b749b1a515e732aa816`。
- 模型语义：`huggingface-modeling-code` 的 `2ff8a4b2752cb54ff8dedfd7408ac7e6b7d2ee40` 中 `src/transformers/models/qwen4_exp/`。当前 HF `main` 没有 Qwen4Exp，不能用其现有 Qwen 系列模型代替。
- 开发基线：Megatron-LM `b4407209223d290bbf7d6f2167d4e2ae51b7a58d`，Megatron-Bridge `1a8b99e05b9aa953c6cf00cc54eea43b50a22857`，TransformerEngine `5e994640625626263aa62250c709e35601563566`。三个仓库在 `feat/qwenair-training-20261008` 分支开发，最终分别提交至 `xshang` fork。
- 验收含义：“正确训练”要求已定义的全部参数具有预期梯度，数值与独立 HF oracle 一致，分布式结果与单卡基线一致，真实规模能完成连续训练、保存与恢复。单个 tiny model backward 只能证明功能路径。

`model-info/p0`、`p1`、`p2` 是有用的本地设计资料，但其中 P1/P2 所称的 QwenAir MCore 代码不在当前 Megatron-LM 工作树内；历史通过数字不能计入本次验收。`QwenAir/QwenAir.pptx` 只有封面，没有技术规格。

## 2. 架构与模块归属

| 目标语义 | 配置值 | 实现归属及做法 |
| --- | --- | --- |
| 文本主干 | H=2560、48 层、上下文 262144、词表 248320 | MCore 新增 QwenAir 独立 config/model；以显式 `layer_types` 决定层类型，保持全局层号；最终投影不共享词嵌入。 |
| 混合层 | 36 GDN + 12 QSA，模式 `[GDN,GDN,GDN,QSA] × 12` | 复用 MCore hybrid/GDN 基础设施，QSA 使用独立模块；不得把普通全注意力或 DSA 当作 QSA。 |
| 四流 HC | 4×H 状态、低秩 320、每层 attention/MLP 各一次 | MCore 专用 grouped zero-centered RMSNorm、read/injection、final mixer；现有 Sinkhorn mHC 公式不同。 |
| GDN | 16 Q/K、48 V heads、维度 128、卷积核 4 | MCore/FLA 路径按 HF 参数形状、QK L2 norm、FP32 decay、sigmoid beta 与输出门控核对；初始化 `A_log=log(U[0.01,16])`。 |
| QSA 主注意力 | 24 Q、2 KV、head dim 256 | MCore 完成 Q/K norm、部分 MRoPE、Q 输出门控；TE 提供可微稀疏注意力实现及 kernel 扩展。 |
| QSA indexer | 4 Q、1 K、dim 128；每 4 token 完整块平均；top 512 块，最多 3 个可见尾 token | 选择须逐 query token 计算，FP32 score、ReLU、causal 与 padding 精确一致。MCore 保存逐 token 索引；TE 接口不能把同一 query block 的选择强制合并。 |
| MoE | 每层 512 routed experts、top 10、每专家中间维 640；另有 gated shared expert | MCore router 采用 FP32 softmax→top-k→入选权重重归一；跨 48 层聚合 router aux（系数 0.001），按 HF 公式核对。 |
| PLE | one-based 第 2 层；二/三元各 8 个 hash head；总表 `[320001536,160]` | MCore 实现 EOS 重置、逐头 hash/offset、gate、dilated depthwise conv；先小表 oracle，后做 TP 分片及流式 checkpoint。真实 BF16 表单独约 95.37 GiB。 |
| MTP | 声明 1 层 QSA+MoE、无 dedicated embedding | 结构与训练目标分开；在训练 shift、loss 系数、共享规则得到权威定义前不得把通用 MCore MTP 默认行为称为 QwenAir 等价。 |
| Vision | 27 层、H=1152、16 heads、patch 2×16×16、merge 2×2 | 后续从 Qwen VL 路径适配，逐层核对 patch/position/merge 与文本注入，再做图像和视频训练。 |

Bridge 负责配置注册、模型实例化与 checkpoint 双向转换；它的当前 MCore pin 是 `5052ce8c…`，与本次独立 MCore 基线不同。Bridge 集成前须明确更新 pin 与依赖锁，并复跑所有测试。TE 负责注意力算子的前反向和后续 Blackwell 优化，不承接模型配置或 indexer 选择。SGLang 的稀疏 kernel 仅有推理前向且数学/布局不完全相同，不能作为训练核直接复用。

## 3. 分阶段开发

### A. 可复现语义和小模型训练

1. 建立 HF 冻结版本的独立 fixture：HC、GDN、PLE、QSA 选择/输出、MoE、完整小文本模型的输出与梯度。按固定公式生成输入并保存 tensor、元数据、源 commit、配置 hash 和 SHA-256；MCore 测试只读取 fixture，不在同一进程调用 HF。
2. MCore 实现可缩小维度/层数/专家数的 QwenAir text model，保留目标配置的拓扑和 HF 参数命名。为 GDN/QSA 交替、PLE one-based 放置、四流更新、router aux、loss/backward 写单元测试。
3. QSA 先用稠密 masked attention 作为数值参考；small BF16 模型执行至少两个 optimizer step，检查有限 loss、主干和 MoE 梯度以及 checkpoint save/reload 后一致性。indexer 在 hard top-k 的 LM loss 下无梯度是预期事实，必须单独记录。
4. Bridge 接入独立模型，检查配置从完整 JSON 读取后的所有重要不变量；合成 checkpoint 双向 roundtrip 必须覆盖融合 QKV/GDN、HC、MoE 和 PLE 小表。真实 checkpoint 尚未提供时不声称真实权重等价。

### B. 稀疏注意力训练与 B200/B300 功能验证

1. TE 先实现逐 token 选块、causal、GQA 的可微稀疏功能路径，用同一索引对稠密 masked oracle 检查输出和 dQ/dK/dV。query chunk 只限制 TE gather 的临时空间；当前 MCore 选择器仍构造 `[S,S]` 可见/选择 mask，必须在生产阶段改为流式或分块索引生成。
2. 在 B200/B300 上测单卡 BF16 forward、backward、优化器步骤，记录 torch/CUDA/cuDNN/NCCL/TE/FLA 版本、GPU 名称、commit 和日志。比较稀疏路径与稠密参考，覆盖短序列、块边界、尾 token、左 padding、top-k ties 和长序列内存。
3. 生产 kernel 需要实际 block-sparse forward/backward 和长上下文显存/吞吐门槛；纯 PyTorch gather 路径仅用于功能验证。只有性能、数值与梯度同时通过，才设为目标训练默认路径。

### C. 并行、真实权重与完整训练

1. 按 TP=1→2→4→8、EP=1→8、PP=1→2/4、CP=1→2 的顺序启用并行；每一档先测试小模型与固定 batch 的 loss/梯度等价，再测试 checkpoint restart。任何新增并行方式都要核对 HC 四流和 QSA indexer 的分片及同步。
2. PLE 大表只在足够的 TP/optimizer sharding 下构造；物理 128 shard 逐片读取并映射到目标 rank，限制峰值内存，校验每片 hash。先完成真实 checkpoint header/key/dtype/shape inventory，再实现 converter；缺少真实 checkpoint 时保留合成 roundtrip 结论。
3. 启用 177B 目标配置，跑多节点连续训练与保存/恢复：记录每步 loss、各模块梯度、峰值显存、吞吐、数值异常、optimizer 状态和重启后 loss 对齐。使用集群共享路径保存代码、数据和日志；Slurm 每节点一个 `srun` task，由 `uv run python -m torch.distributed.run` 启动各 GPU worker。
4. 接入 Vision 及 image/video fixture 后，分别验证视觉主干、跨模态输入拼接、端到端 loss/backward 与目标规模训练。

### D. 训练契约关闭

HF Qwen4Exp 的 hard top-k 没有 indexer 的 LM 梯度；其测试明确说明 indexer 由**独立目标**训练，但没有公开 teacher、label、loss 形式、系数与 backward golden。配置声明 MTP，HF 的 Qwen4Exp forward 没有 MTP 训练实现，也未定义 shift/loss/detach/共享权重规则。获得模型方批准的这些规范和 oracle 后，再实现对应 loss、验证参数梯度并开展完整预训练验收。在此之前可验证主干 LM 训练路径，不能宣称 QwenAir 的所有训练目标已正确实现。

## 4. 验收证据与提交

每个阶段保存：三仓库 commit、依赖 lock/container digest、完整 Slurm 命令及作业 ID、输入与 fixture hash、每 rank 日志、数值比较、grad 检查、optimizer step、checkpoint restart 与显存/性能结果。测试失败需定位首个非 NCCL traceback，再修正实现并重跑相关门槛。先在单卡 B200/B300 完成功能验证，再扩展到 8 卡和多节点。

各仓库各自提交最小可审查改动并推送 `git@github.com:shangxiaokang/{Megatron-LM,Megatron-Bridge,TransformerEngine}.git` 的工作分支。Megatron-LM 提交按仓库要求同时使用 `-s -S`；对应代码与测试在各自仓库内，避免把参考仓库的已有本地改动混入提交。最终完成状态以运行证据与上述所有门槛为准。
