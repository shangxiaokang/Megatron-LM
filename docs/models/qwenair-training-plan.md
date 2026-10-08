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

Bridge 负责配置注册、模型实例化与 checkpoint 双向转换；本工作分支已把 MCore submodule pin 更新到 QwenAir 首次实现提交 `9af44dff`，并让该 submodule 指向 xshang fork。Bridge 的全局 TE pin 尚未改变；`te_reference` 需要单独安装或注入本次 TE fork 的 QSA API，不能把这种临时联调视为完整 TE fork 安装验证。TE 负责注意力算子的前反向和后续 Blackwell 优化，不承接模型配置或 indexer 选择。SGLang 的稀疏 kernel 仅有推理前向且数学/布局不完全相同，不能作为训练核直接复用。

以完整模型约 177,392,830,576 参数计，单独 BF16 权重约 330.42 GiB、BF16 梯度约 330.42 GiB、两份 FP32 Adam 动量约 1321.68 GiB，合计约 1982.52 GiB（12 byte/参数）。这还不含 FP32 master weight、activation、通信缓冲和 checkpoint 峰值；理想平均分到 8 GPU 也约 247.81 GiB/GPU。目标训练必须有跨设备参数/梯度/优化器分片或 offload，并以实际 B200/B300 显存留出激活与通信空间。PLE 表单独约 95.37 GiB BF16，不能在每个 rank 全量复制。

当前 `max_single_rank_parameters=100_000_000` 和 `max_single_rank_ple_elements=50_000_000` 是参考模型的安全限额，目标形状即便启用专家/PLE 分片仍会被拒绝。例如 512-way EP 时，仅 routed experts 平均每 rank 也约 236M 参数。真实配置必须先选定 TP/EP/PP、优化器分片与 checkpoint 策略，按实测显存显式提高限额；提高限额本身不代表已经具备内存可行性。

## 3. 分阶段开发

### A. 可复现语义和小模型训练

1. 建立 HF 冻结版本的独立 fixture：HC、GDN、PLE、QSA 选择/输出、MoE、完整小文本模型的输出与梯度。按固定公式生成输入并保存 tensor、元数据、源 commit、配置 hash 和 SHA-256；MCore 测试只读取 fixture，不在同一进程调用 HF。
2. MCore 实现可缩小维度/层数/专家数的 QwenAir text model，保留目标配置的拓扑和 HF 参数命名。为 GDN/QSA 交替、PLE one-based 放置、四流更新、router aux、loss/backward 写单元测试。
3. QSA 先用稠密 masked attention 作为数值参考；small BF16 模型执行至少两个 optimizer step，检查有限 loss、主干和 MoE 梯度以及 checkpoint save/reload 后一致性。indexer 在 hard top-k 的 LM loss 下无梯度是预期事实，必须单独记录。
4. Bridge 接入独立模型，检查配置从完整 JSON 读取后的所有重要不变量；合成 checkpoint 双向 roundtrip 必须覆盖融合 QKV/GDN、HC、MoE 和 PLE 小表。真实 checkpoint 尚未提供时不声称真实权重等价。

### B. 稀疏注意力训练与 B200/B300 功能验证

1. TE 先实现逐 token 选块、causal、GQA 的可微稀疏功能路径，用同一索引对稠密 masked oracle 检查输出和 dQ/dK/dV。MCore 的无 padding TE 路径已改为按 query chunk 生成索引，不再分配 `[S,S]` 选择 mask；目前每个已完成块的 query 组仍单独调用 top-k，计算量和 kernel launch 数随上下文二次/线性上升，目标长度必须另做融合候选生成与 top-k kernel。
2. 在 B200/B300 上测单卡 BF16 forward、backward、优化器步骤，记录 torch/CUDA/cuDNN/NCCL/TE/FLA 版本、GPU 名称、commit 和日志。比较稀疏路径与稠密参考，覆盖短序列、块边界、尾 token、左 padding、top-k ties 和长序列内存。
3. 生产 kernel 需要实际 block-sparse forward/backward 和长上下文显存/吞吐门槛；纯 PyTorch gather 路径仅用于功能验证。只有性能、数值与梯度同时通过，才设为目标训练默认路径。

### C. 并行、真实权重与完整训练

1. 按第 6 节的并行/优化器/内存依赖顺序推进；每一档先测试小模型与固定 batch 的 loss/梯度等价，再测试 checkpoint restart。TP、PP、CP、EP 不是互不相关的开关，进程组、参数所有权和 router aux 必须一起定义。
2. PLE 大表只在足够的表分片、优化器分片或 offload 下构造；物理 128 shard 是 checkpoint 文件布局，不能直接当作运行时表进程组大小。逐片读取并映射到目标 rank，限制峰值内存，校验每片 hash。先完成真实 checkpoint header/key/dtype/shape inventory，再实现 converter；缺少真实 checkpoint 时保留合成 roundtrip 结论。
3. 启用 177B 目标配置，跑多节点连续训练与保存/恢复：记录每步 loss、各模块梯度、峰值显存、吞吐、数值异常、optimizer 状态和重启后 loss 对齐。使用集群共享路径保存代码、数据和日志；Slurm 每节点一个 `srun` task，由 `uv run python -m torch.distributed.run` 启动各 GPU worker。
4. 接入 Vision 及 image/video fixture 后，分别验证视觉主干、跨模态输入拼接、端到端 loss/backward 与目标规模训练。

### D. 训练契约关闭

HF Qwen4Exp 的 hard top-k 没有 indexer 的 LM 梯度；其测试明确说明 indexer 由**独立目标**训练，但没有公开 teacher、label、loss 形式、系数与 backward golden。配置声明 MTP，HF 的 Qwen4Exp forward 没有 MTP 训练实现，也未定义 shift/loss/detach/共享权重规则。获得模型方批准的这些规范和 oracle 后，再实现对应 loss、验证参数梯度并开展完整预训练验收。在此之前可验证主干 LM 训练路径，不能宣称 QwenAir 的所有训练目标已正确实现。

## 4. 验收证据与提交

每个阶段保存：三仓库 commit、依赖 lock/container digest、完整 Slurm 命令及作业 ID、输入与 fixture hash、每 rank 日志、数值比较、grad 检查、optimizer step、checkpoint restart 与显存/性能结果。测试失败需定位首个非 NCCL traceback，再修正实现并重跑相关门槛。先在单卡 B200/B300 完成功能验证，再扩展到 8 卡和多节点。

各仓库各自提交最小可审查改动并推送 `git@github.com:shangxiaokang/{Megatron-LM,Megatron-Bridge,TransformerEngine}.git` 的工作分支。Megatron-LM 提交按仓库要求同时使用 `-s -S`；对应代码与测试在各自仓库内，避免把参考仓库的已有本地改动混入提交。最终完成状态以运行证据与上述所有门槛为准。

## 5. 当前实测进度（2026-10-08）

- 冻结 HF 的 HC、PLE、GDN、QSA 前向与 MCore 小模型进行独立权重/输出对照；目标文本参数数目及 PLE hash 与静态 oracle 一致。此项不代替完整模型的梯度 golden。
- B300 上使用 PyTorch `2.9.0a0+145a3a7bda.nv25.10`、CUDA 13.0 与 NVRX 0.6.0：MCore 初始单卡 12 项、TE 原始 QSA 134 项、后续 TE 全部定向 188 项以及 MCore↔TE 接口 3 项测试通过。TE FP32 严格对照需要在完整前向和反向期间禁用 TF32；BF16 仍可运行。
- B300 SXM6 AC 作业 `4799604` 中，`examples/qwenair/train_reference.py` 已用 dense 和 `te_reference` 各完成两个 BF16 autocast/AdamW 步骤。每个已实现的主干模块均有非零有限梯度；checkpoint 恢复后第二步 loss、所检查参数的梯度、模型权重与 AdamW 状态逐项完全重放。hard top-k indexer 的 LM 梯度为空符合当前公开实现，但独立训练目标未定义。
- 该集群镜像中的 TE 原生扩展早于本次 TE fork；联调通过绝对路径加载新的纯 Python QSA 函数并注入镜像内已安装的 TE 包。必须用匹配的扩展重建或更新镜像，才能称为 TE fork 整包验证。
- B300 双卡作业 `4799432` 中，PLE 2-rank 的变长查表、反向以及 MCore 分布式 checkpoint 实际写入/恢复 2/2 通过；CPU Gloo 对照亦通过。当前仅 PLE 表按行分片，专家并行、长上下文选择器与完整多模态训练尚未实现。
- 新的 TE indexed SDPA 路径在 B300 上 51/51 定向用例通过，但 profiler 显示布尔选择 mask 令它回退到 PyTorch math SDPA，强制 FlashAttention 会拒绝非空 mask。因此该路径是功能验证方案，尚无目标上下文长度的吞吐/显存验收，不能作为生产稀疏 kernel。
- B300 作业 `4800005` 对流式无 padding QSA 选择器、超出 dense 限额的 TE 路径及完整小文本 TE 训练共 3 项通过；作业 `4800095` 对 MCore 的 dense、TE gather 与 indexed SDPA 接线、视觉 `inputs_embeds`/原始 PLE ID 接口等 23 项 reference 回归通过。`te_indexed_sdpa` 仍需动态注入当前 TE fork 的 Python API，且实际 SDPA 后端为 math。
- 独立 MoE EP 模块在 2×B300 NCCL 作业 `4800048` 3/3 通过，覆盖 MCore all-to-all dispatcher、局部 packed experts、router/shared 梯度、全局 aux 和分布式 checkpoint。
- 2×B300 NCCL 作业 `4800461` 中，完整小文本模型已联合 PLE 行分片与 MoE expert parallel 完成 1/1 端到端测试：GDN/QSA/HC、变长 rank-local token、全局 CE 与 router aux、与未分片参考的逐参数梯度/更新，以及模型 DCP 加 rank-local AdamW 状态的第二步 BF16 精确重放。此实现要求 PLE 与 EP 使用同一进程组，expert TP=1；复制参数由原型训练循环显式 SUM 梯度。大规模 DDP/优化器分片及 grouped expert GEMM 尚未接入。
- B300 作业 `4800570` 在 EP 接线后重新执行单卡 reference/TE/indexed SDPA 回归 23/23 通过；作业 `4800615` 的独立双卡 MoE 测试 3/3 通过，包含一个 rank 没有本地 token 时的 all-to-all 与梯度对照。完整文本模型仍要求训练器为每 rank 提供非空 local batch；batch=0 的全模型路径尚未验收。
- B300 作业 `4800864` 中，Bridge 的冻结 HF image/video oracle、视觉 wrapper 与文本映射共 11/11 通过；MCore 使用持久 HF fixture 验证视觉 feature scatter、三轴位置、PLE 原始 token ID 及 CE backward 共 2/2 通过。
- Transformer Engine fork `c4f14012` 已在 PyTorch 25.10/CUDA 13.0 镜像中针对 B300 `sm_103a` 完成原生 wheel 构建，并从隔离安装目录加载对应 `.so`。B300 作业 `4801923` 使用该原生包运行 MCore QwenAir reference/TE/indexed SDPA 回归并以 `0:0` 完成；2×B300 作业 `4801872` 将 `te_indexed_sdpa` 与 PLE+EP 完整小模型的两步 BF16/AdamW、DCP 与 optimizer restart 联合验证并以 `0:0` 完成。indexed SDPA 仍为 math backend，原生构建成功不等于生产稀疏 kernel 已实现。

## 6. 距离 177B 目标训练的工程差距与下一阶段门槛

以下均针对当前小模型原型，而非已通过的目标规模能力。当前 `QwenAirForCausalLM` 是整个 48 层在每个 rank 构造的 `MegatronModule`，只接受显式的 PLE/EP/ETP 进程组；`QwenAirTextConfig.validate()` 仍拒绝 TP、PP、CP、EP、ETP 的非 1 配置值。两卡 EP 测试绕过配置字段，使用同一组做 PLE 表行分片和 expert 分片，TP=PP=CP=1，训练循环手工 SUM 复制参数梯度，优化器为每 rank 普通 AdamW。这验证了数学和局部分片，不等于 MCore 多维训练入口。

### 6.1 所有权、计算与内存缺口

1. **并行网格与梯度所有权。** 将 QwenAir 模型接到 MCore `ProcessGroupCollection`，区分 TP、PP、CP、EP、专家 TP、普通 DP、专家 DP 和 PLE 表分片/副本组。标准 MCore DDP 以 `param.allreduce=False` 将专家权重送入专家 DP 桶；现有 QwenAir packed experts 和 PLE 表没有完成这个参数标记/桶归属，也没有接入 `DistributedDataParallel`、`finalize_model_grads` 或 `DistributedOptimizer`。EP rank 分担同一全局 loss 时，复制参数需跨 EP **SUM** 局部梯度；独立 DP 副本才做平均或等价归一化。PLE 行 shard 和 expert shard 不得沿其所有权分片组再 all-reduce。当前 `sync_ep_replicated_gradients()` 仅是单独测试辅助，不能与 DDP 重复调用。
2. **TP。** 词嵌入/LM head、GDN 融合 QKV 和卷积、QSA Q/K/V 与输出、HC 的 4H read/injection/mixer、shared expert、router 都仍用未分片 `nn.Linear`/`nn.Embedding`。需要采用 MCore `VocabParallelEmbedding`、`ColumnParallelLinear`、`RowParallelLinear` 和 vocab-parallel CE，或具有相同前反向/检查点契约的包装；保留 HF 逻辑参数名与投影切片。目标 QSA 只有 2 个 KV head、indexer 只有 1 个 K head：TP>2 不能朴素按 head 均分 KV；需复制 KV、定义副本梯度归约，并让所有 TP rank 使用完全相同的逐 token 选择。indexer 的 4 个 Q head 若分片，其跨 head 得分和 top-k 必须先全局规约；初期可复制整个 indexer，随后再优化。GDN 的 16 QK/48 V head 及四流 HC 也要逐项核对 TP 切片。
3. **PP 与全层损失。** 当前每 rank 创建所有层、输入 embedding 和 LM head，没有 `pre_process/post_process`、本地全局层号、`set_input_tensor` 或 MCore pipeline schedule。PP 阶段必须只拥有对应层；PLE 固定在全局第 2 层，LM head/final HC mixer 仅在末 stage。HF router aux 对 48 层和有效 token 统一计数；现有 loss 仅对本 rank 已构造层求值。PP 拆分后必须在不破坏 1F1B 微批次顺序的前提下交换可微概率和不可微计数，并保持与未切分全层 loss/梯度相同。
4. **CP 与长序列状态。** GDN 现为逐 token Python FP32 recurrent loop，没有分块反向、activation checkpoint 或跨 CP shard 的递推状态/卷积 halo。MCore 既有 GDN 的 chunkwise/headwise CP 与 FLA kernel 可作为候选，但先要证明它与冻结 HF 的参数布局和数值一致。QSA 当前 indexer 虽按 query chunk 限制临时内存，仍扫描全部已完成块，计算量随上下文二次增长；TE `te_reference`/`te_indexed_sdpa` 仍需完整本地 K/V，后者在布尔 mask 下实测退化为 math SDPA，尚无跨 CP rank 取选中块和生产级 block-sparse backward。PLE 的 EOS-aware ngram、depthwise conv、GDN causal conv、MRoPE 位置及 next-token label 都需要 CP 边界 halo/状态。不能把标准全注意力的 CP 开关直接用于这一混合层。
5. **专家计算与表访问。** 目前 MoE 用 MCore all-to-all dispatcher，但 `QwenAirExperts.forward_dispatched()` 对每个本地专家执行 Python 循环、`tokens_per_expert.tolist()` 和单独 GEMM；512 experts 目标下不能据此推断吞吐。应优先适配 MCore/TE `TEGroupedMLP` 或等价 grouped GEMM，并以 packed HF `[E,2I,H]`、`[E,H,I]` 的 SwiGLU 顺序做双向权重/梯度映射。PLE lookup 已可变长 all-to-all，但每次取 host split、查表通信量和真实表存储尚未压测；表的 optimizer state 还未按所有权分片。
6. **checkpoint/Bridge。** 当前 `sharded_state_dict` 用一个进程组描述 PLE 与 expert 的 axis-0 shard，强制两者是同一 `ProcessGroup`；其他权重按复制张量记录。不同 PLE/EP/TP 组、PP 层偏移、CP/DP 副本 ID、普通/专家/表的 optimizer state 和跨拓扑 reshard 均未定义。Bridge provider 仍拒绝非单 rank TP/PP/EP/CP，DirectMapping 只支持逻辑张量的单卡往返，没有真实 128 文件 PLE 的流式装载/导出，也未启用 HF pretrained export。物理 128 文件 shard 与运行时进程组大小应独立映射。

以 `model-info/p0/parameter-ledger.json` 的条件性参数清单核算：文本 176,943,899,520、vision 448,931,056，合计 177,392,830,576，尚不包括需确认训练契约的 MTP 额外参数。其中 routed experts 为 `48×512×3×640×2560 = 120,795,955,200`，PLE 表为 `320001536×160 = 51,200,245,760`，两者之外还有约 5.397B 参数（含 vision）。仅按 512-way EP/表分片，**每 rank** 仍持有 235,929,600 个 expert 参数和 100,000,480 个 PLE 参数；这已分别超过当前 100M/50M 安全阈值。按 BF16 权重、BF16 梯度和两份 FP32 Adam moment 的 12 byte/参数下界，两项本地 shard 分别约 2.64/1.12 GiB，尚未算其余权重、FP32 master、activation、临时 gather 与通信缓冲。单样本、262144 token 的一个 `[B,S,4H]` BF16 四流隐藏张量就是 5.0 GiB。PP、TP、CP、activation recompute、optimizer 分片/offload 及峰值内存实测缺一不可；仅提升配置安全阈值不解决训练内存。

MCore 网格不能按 `TP×PP×CP×EP×DP` 盲目相乘：其普通模型满足 `world_size=TP×PP×CP×DP`，专家侧满足 `world_size=ETP×EP×PP×expert_DP`，EP 从既有 rank 布局中组织。每个测试先打印并核对实际 `ProcessGroupCollection` 成员、参数所有者和 loss 归一化，再运行训练。下列作业规模是最低验收配置，不表示性能达标。

### 6.2 可执行开发顺序及最小 B300 验收

1. **拓扑/资源清单。** 增加不分配目标权重的 177B 参数、每组每 rank 参数/梯度/Adam/激活/通信预算器；先把 QwenAir config、模型工厂和 Bridge provider 的显式并行布局、`ProcessGroupCollection` 与 fail-closed 约束接通。逐参数声明 logical shape、TP/EP/PLE 轴及副本组，定义空 local batch 的全组一致拒绝规则。验收：2×B300、tiny EP2 与目标 config 的 *dry-run*；每 rank 输出一致的网格/预算，错误拓扑在构造前报错，不尝试分配 177B。
2. **Grouped expert 计算。** 先在现有 EP2 all-to-all 后接 MCore/TE grouped expert kernel，消除每专家 Python GEMM/host token count；为 SwiGLU packed 权重建立 HF↔kernel 逻辑映射，保留空专家/空源 rank、top-k 重复目的地和 BF16 autocast 语义。验收：2×B300，4/16 experts 的 FP32/BF16 前向、输入/权重/router 梯度对照及 DCP 重启；再 8×B300 以 64 experts/EP8 测 token 分布极不均匀、峰值显存和 profiler，确认专家计算不随本地专家数线性增长 Python launch 数。
3. **DDP 与优化器所有权。** 标记 expert shard 走 `expt_dp`、复制参数走适当 `dp_cp`/EP SUM、PLE 表走其独立副本组；接 MCore DDP/`DistributedOptimizer`，去掉训练循环手工 `sync_ep_replicated_gradients()`。验收：4×B300，EP2 加两个专家副本的组布局；比较与当前无分片 FP32 oracle 的 global CE/router aux、每类参数梯度、两步 BF16 更新；保存模型与**分布式 optimizer state** 后换进程重启，第二步 loss、权重、optimizer 状态一致。另测某 rank 无有效 CE label 与非均匀 token 数，排除 SUM/AVERAGE 缩放错误。
4. **PLE 独立表组与跨组 checkpoint。** 解除 PLE group 必须与 EP group 对象相同的限制，为每个表 shard、expert shard、将来的 TP 权重和普通副本构造独立 `ShardedTensor`/replica metadata；先用小表实现 group 重排与 optimizer state。验收：4×B300 构造 PLE2×EP2 不同组、非均匀 lookup，比较输出/梯度并跨相同拓扑重启；8×B300 增加 DP2 并做 2→4 table shard reshard。用合成 128 文件 manifest 验证逐片流式转换、hash、峰值内存上界；真实文件到位后逐片重测，不能把小表 DCP 当作真实权重转换。
5. **TP 与四流。** 从 TP2 开始接 embedding/LM head、HC、GDN、QSA、router、shared expert，再处理 TP4/8 的 KV/indexer 复制和同步。每个算子保留 HF logical checkpoint 切片，先不做 CP/PP。验收：2×B300 TP2/EP1/ETP2 对单卡 tiny reference；4×B300 TP2/EP2/ETP2 对现有两卡 EP oracle，检查逐层输出、全局 top-k、梯度与 DCP；8×B300 TP4/EP2/ETP4 覆盖 2 KV heads/1 indexer K head 不可整除的目标几何及 optimizer restart。
6. **PP 与跨层 aux。** 按全局层号切 48 层，只在首/末 stage 建 embedding/final mixer/head，把 router 统计和梯度正确带过 pipeline 微批次；嵌入第 2 层 PLE 与 GDN/QSA 周期不因 stage 边界变化。验收：4×B300 PP2/EP2/ETP1，4 或 8 层 tiny 全层 HF oracle，1F1B 两个以上 microbatch 的逐层输出、CE+全层 aux 梯度和模型+optimizer 重启；再 8×B300 TP2/PP2/EP2/ETP2 测流水线通信与 checkpoint key/offset。
7. **长序列 kernel 与 CP。** 先把 GDN 的 causal conv+递推替为经 HF 数值/梯度验证的分块 FLA/TE 实现，把 QSA 选择器做块摘要 GEMM/在线 top-k，TE 注意力做真正 block-sparse forward/backward，避免 math SDPA fallback 与 `S²` 临时量。然后设计 CP 边界：GDN 递推状态和卷积 halo、PLE ngram/EOS/conv halo、QSA 全局 block ID/远端 K/V、位置和 label。验收：2×B300 CP2 的 GDN+QSA+PLE，小序列专门跨 EOS、4-token block、卷积边界做 FP32/BF16 输出/梯度对照与重启；再在 8×B300 上逐级测 16k→64k→262144 token 的一层 forward/backward，记录每 rank 峰值显存、吞吐、kernel 占比和无 math fallback。若目标长度单层都无法过门槛，不进入 48 层训练。
8. **组合网格与文本规模。** 使用第 1 步确定的实际 MCore rank 布局，先 16×B300 测 TP2/PP2/CP2/DP2、EP2/ETP2/expert-DP2 与 optimizer sharding，检查与小模型基线一致，再扩至能容纳 176.944B 文本参数的多节点网格。验收：目标文本配置 BF16 连续训练多个 step，所有**已定义** loss 与参数梯度有限、无 OOM，单步预算与 profiler 可解释；跨节点保存完整模型/optimizer/RNG/data 位置后重启，下一步 loss/梯度/参数精确或在预设 BF16 容差内重放；记录扩展效率。若 indexer 独立目标或 MTP 规范仍缺失，只能称为主干 LM 训练验收，不能称为完整 QwenAir 预训练。
9. **Vision 与完整目标。** 在文本阶段通过后接入 0.449B 参数的 vision trunk、位置/patch/merge、image/video 输入分配与文本 embedding 注入，逐层对照冻结 HF；随后在同一训练网格中验证图文/视频 batch、路由负载与 checkpoint。验收：2×B300 小图像/短视频的融合输出、loss、文本及 vision 梯度和 optimizer 重启与 HF oracle 一致；再按目标配置做多节点 177.393B 完整模型训练及跨节点恢复。独立 indexer 目标、MTP loss/共享与权重绑定须得到权威规范和 golden 后另设数值门槛；规范缺失时，即使图文主干 LM 可训练，也不宣称完整预训练收敛或目标等价。

以上每步必须保存测试参数、组成员、镜像/代码 commit、Slurm job ID、各 rank 日志、tensor 对照、checkpoint manifest、峰值显存与失败首个 traceback。上游 MCore/TE 有可复用组件，但要通过 QwenAir 自身的 HF 数值和梯度门槛后才能替换当前参考实现。
