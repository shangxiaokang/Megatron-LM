# QwenAir 训练实现方案与验收计划

状态：最终 MCore+Bridge+TE 组合已通过 8×B200 text base-LM 功能、保存与独立进程恢复门槛；32×B300 目标文本形状作业仍在等待资源，完整 QwenAir 预训练仍在开发中（2026-10-08）。本文件列出已经取得的运行证据、当前支持边界及后续完成条件；任何阶段只有取得对应运行证据后才可标记完成。

## 1. 冻结输入与目标

- 目标配置：`configs-and-numbers/sglang/agg/accuracy/numbers/provenance/bf16-model-config.json`，SHA-256 `b7d4f14b5891c998e767f92637e1f9c81eca57252f645b749b1a515e732aa816`。
- 模型语义：`huggingface-modeling-code` 的 `2ff8a4b2752cb54ff8dedfd7408ac7e6b7d2ee40` 中 `src/transformers/models/qwen4_exp/`。当前 HF `main` 没有 Qwen4Exp，不能用其现有 Qwen 系列模型代替。
- 当前功能代码：Megatron-LM `70e662aac0ea4eb53fbf60b1c105f278ea41c8e4`、Megatron-Bridge `431f2f17540af31a7fa40970f5a28be1bb2b7716`、TransformerEngine `3250741db1e06acd638da6124bc193405565f771`。三个仓库均在 `feat/qwenair-training-20261008` 分支。Bridge `431f2f175` 的 MCore gitlink 指向 `70e662aac`，MCore 与 Bridge 的依赖锁均固定到 xshang TE `3250741d`；以下证据逐项注明实际受测提交，不能把中间提交上的结果自动归到最终代码点。
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

Bridge 负责配置注册、模型实例化与 checkpoint 双向转换。当前分支把 Flux/Wan recipe 改为按需导入，使 QwenAir recipe 不依赖 diffusion extras；provider 接受显式 EP/EDP 进程组，并在 TP/PP/CP/ETP 非 1 时 fail-closed。Bridge `431f2f175` 还显式传递 `linear_conv_kernel_dim`、`linear_key_head_dim`、`linear_value_head_dim`、`linear_num_key_heads`、`linear_num_value_heads`，避免 `TransformerConfig` 的默认 GDN 几何覆盖 QwenAir 配置。为匹配 Bridge 的训练与 checkpoint import，MCore 分支还补齐 FSDP V1/V2 配置入口、可区分的 FSDP V2 标记、pre-GTP capability shim，以及 tokenizer asset 保存和 `raise_on_error` 传播接口。这些兼容入口不代表 QwenAir 已实现 generalized tensor parallelism。Bridge recipe 选择 `te_triton`、显式配置同步 DDP `bucket_size=40_000_000` 并令 `log_interval=1`。目标文本 bring-up recipe 的默认 sequence length 已从 4096 降到 64：reference GDN 的逐 token recurrence 会为 backward 保留 FP32 state，按 36 个 GDN 层估算，4096 token 时仅该 state 就约 432 GiB/GPU 下界；64 token 仍只是短序列 bring-up。TE 负责注意力算子的前反向和 Blackwell 原型优化，不承接模型配置或 indexer 选择。SGLang 的稀疏 kernel 只有推理前向且数学/布局不完全相同，不能作为训练核直接复用。

以完整模型约 177,392,830,576 参数计，单独 BF16 权重约 330.42 GiB、BF16 梯度约 330.42 GiB、两份 FP32 Adam 动量约 1321.68 GiB，合计约 1982.52 GiB（12 byte/参数）。这还不含 FP32 master weight、activation、通信缓冲和 checkpoint 峰值；理想平均分到 8 GPU 也约 247.81 GiB/GPU。目标训练必须有跨设备参数/梯度/优化器分片或 offload，并以实际 B200/B300 显存留出激活与通信空间。PLE 表单独约 95.37 GiB BF16，不能在每个 rank 全量复制。

当前 `max_single_rank_parameters=100_000_000` 和 `max_single_rank_ple_elements=50_000_000` 是分配前资源保护阈值，而不是 EP 功能限制。`TP=PP=CP=ETP=1`、`world_size=EP×EDP` 的布局已经接通；目标形状需要按所选 EP 显式提高保护阈值，并先运行 allocation-free dry-run。即使 512-way EP，仅 routed experts 平均每 rank 仍约 236M 参数；提高阈值只允许构造，不代表已有足够显存或达到了训练吞吐要求。

## 3. 分阶段开发

### A. 可复现语义和小模型训练

1. 建立 HF 冻结版本的独立 fixture：HC、GDN、PLE、QSA 选择/输出、MoE、完整小文本模型的输出与梯度。按固定公式生成输入并保存 tensor、元数据、源 commit、配置 hash 和 SHA-256；MCore 测试只读取 fixture，不在同一进程调用 HF。
2. MCore 实现可缩小维度/层数/专家数的 QwenAir text model，保留目标配置的拓扑和 HF 参数命名。为 GDN/QSA 交替、PLE one-based 放置、四流更新、router aux、loss/backward 写单元测试。
3. QSA 先用稠密 masked attention 作为数值参考；small BF16 模型执行至少两个 optimizer step，检查有限 loss、主干和 MoE 梯度以及 checkpoint save/reload 后一致性。indexer 在 hard top-k 的 LM loss 下无梯度是预期事实，必须单独记录。
4. Bridge 接入独立模型，检查配置从完整 JSON 读取后的所有重要不变量；合成 checkpoint 双向 roundtrip 必须覆盖融合 QKV/GDN、HC、MoE 和 PLE 小表。真实 checkpoint 尚未提供时不声称真实权重等价。

### B. 稀疏注意力训练与 B200/B300 功能验证

1. TE 已实现逐 token 选块、causal、GQA 的可微稀疏 reference/indexed 路径和单次 forward、单次 backward launch 的 Triton 原型；使用同一索引对稠密 masked oracle 检查输出和 dQ/dK/dV。MCore 的无 padding selector 按 query chunk 生成索引，不分配 `[S,S]` 选择 mask。Triton 原型消除了逐 query-block attention dispatch，但 selector 候选生成和 top-k 仍需面向 262K 上下文继续融合、分块和跨 CP 设计。
2. 在 B200/B300 上测单卡 BF16 forward、backward、优化器步骤，记录 torch/CUDA/cuDNN/NCCL/TE/FLA 版本、GPU 名称、commit 和日志。比较稀疏路径与稠密参考，覆盖短序列、块边界、尾 token、左 padding、top-k ties 和长序列内存。
3. `te_triton` 已有实际 indirect K/V block-sparse forward/backward 原型和短/中序列基准；它仍是 prototype。只有 262K selector、远端 K/V、显存与吞吐门槛同时通过后，才可称为最终生产 kernel。

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

当前功能代码点为 MCore `70e662aac`、Bridge `431f2f175`、TE `3250741d`，最终精确组合已由作业 `4810004` 和 `4810199` 验证。早期 Bridge 多卡证据来自 Bridge `622681085`、MCore `830089a60`、TE `3250741d`；该 Bridge provider 未传递五个 `linear_*` GDN 字段，因此当时训练实际采用 `TransformerConfig` 默认的 16/32 个 key/value heads 和 128 维，而不是 tiny fixture 声明的 GDN 几何。最终 Bridge `431f2f175` 在显式分桶改动之上修复这些字段，并把目标 recipe 默认 sequence length 降为 64。目标 32 卡真实形状仍没有通过记录。

- 冻结 HF 的 HC、PLE、GDN、QSA、MoE 与小文本模型 fixture 已用于输出、梯度、参数映射和视觉输入 seam 的独立对照。目标文本参数数目及 PLE hash 与静态 oracle 一致；这些 fixture 不包含未公开的 indexer 独立训练目标或 MTP 训练契约。
- MCore 已接入显式 `ProcessGroupCollection`、MCore DDP、`DistributedOptimizer` 和 distributed checkpoint。dense 参数通过 WORLD 同步；routed expert 与 PLE row shard 设置 `allreduce=False`，只在持有同一 local shard 的 EDP ranks 间同步。全局 CE 按 WORLD 有效 token 数加权，router aux 按 EDP 语义归一化。
- 4×B300 作业 `4804756` 与 8×B200 作业 `4805178` 均通过 dense QSA 的多卡 BF16 base-LM gate，覆盖有限 loss/grad norm、distributed Adam 更新、模型与 optimizer DCP 保存/加载以及 restart 重放。这证明当前 `TP=PP=CP=ETP=1`、`EP×EDP` 路径可运行；它不证明目标 176.944B 文本模型或 262K 上下文已经训练。
- allocation-free 作业 `4805248` 使用冻结目标 JSON 验证 `world=EP=512`、`EDP=1`，逻辑文本参数为 `176,943,899,520`。每 rank 估算 replicated `4,947,698,560`、PLE `100,000,480`、routed expert `235,929,600` 个参数；存储下限为 BF16 参数 9.842 GiB、FP32 gradient 19.683 GiB、分片 Adam master/m/v 3.862 GiB，总计 33.387 GiB/rank。该估算不含 activation、workspace、fragmentation 与 checkpoint staging；`EP=511` 在分配前 fail-closed。
- 4×B300 作业 `4805736` 验证 MCore `6c8d63320` 的 shard 初始化和 DDP bucket 报告：expert 按 global expert ID、PLE 按逻辑表元素区间获得与 runtime shard 边界无关的确定性初始化；调用方 global RNG 不被消耗。报告会拒绝达到 `2^31` elements 的单次 collective。同步 DDP 的最终分桶语义由下述 `4809119` 覆盖。
- TE `3250741d` 提供 `qsa_block_sparse_attention`、`qsa_indexed_sdpa_attention` 与 `qsa_triton_attention` 的兼容 public API。作业 `4804303` 完成 201/201 correctness，并测得完整 forward+backward 中位数：S=512 时 indexed 为 129.385 ms、17,531,904 B、64 dispatch，Triton 为 1.803 ms、17,351,680 B、1 dispatch；S=4096 时 indexed 为 1020.478 ms、138,675,200 B、512 dispatch，Triton 为 10.640 ms、138,806,272 B、1 dispatch。
- TE 作业 `4805785` 通过 public API 兼容检查和目标 QSA `24:2` GQA 几何的 BF16 forward、dQ、dK、dV backward。该作业不是 benchmark；上述性能数字来自 `4804303`。Triton 实现仍是短/中序列 prototype，不能据此推断 262K、CP 或多节点远端 K/V 的性能与容量。
- 4×B200 作业 `4806499` 使用 MCore `a37c60dba` 和 TE `3250741d`（节点容器的匹配 native extension 加该提交的 QSA Python/Triton 实现）并以 `COMPLETED 0:0` 结束。初始化测试 5/5 通过；router audit 覆盖全部 4 层、每层 64 个真实 token，并观测到 6 组不同 top-k expert 选择，排除了旧实现的全零 router 固定选择。分布式定向测试 7/7、TE 定向测试 6/6 通过。`te_triton` 以 EP2×EDP2 完成两步 distributed Adam，loss `4.183996 → 4.164544`、gradient norm `0.649485 → 0.594810`，并在四片 DCP 上完成 model/optimizer exact restart，输出 `QWENAIR_TRITON4_DCP_RESTART_PASS`。
- Bridge preflight 作业 `4808610` 固定 Bridge `622681085`、MCore `830089a60`、TE `3250741d`，以 `COMPLETED 0:0` 结束。它验证实际 `megatron.bridge.training.pretrain` import、lazy recipe import、真实 `NullTokenizer.save_tokenizer_assets(..., raise_on_error=True)`、B200 native TE overlay 和精确 QSA Python 源；Bridge 定向测试 18/18、MCore 定向测试 13/13 通过，router 初始化非零且观测到 225 组不同 expert 选择，输出 `QWENAIR_BRIDGE_TRITON_PREFLIGHT_PASS`。当时的测试没有断言五个 `linear_*` 字段，因此该作业只能作为 import、API 和执行前检查证据，不能作为 GDN 结构正确证据。
- 8×B200 作业 `4808736` 使用同一组精确提交，以 EP4×EDP2、BF16、`te_triton`、distributed Adam 运行并以 `COMPLETED 0:0` 结束。iteration 1 的 TensorBoard 指标为 LM loss `4.852964878`、router aux `2.000883818`、gradient norm `0.415851295`；新进程自动发现 `iter_0000001` 并恢复 model 与 `dp_reshardable` distributed optimizer，iteration 2 指标为 LM loss `4.851634979`、router aux `2.000961304`、gradient norm `0.4181646705`，tracker 最终为 2，输出 `QWENAIR_BRIDGE_TRITON_SAVE_RESUME_PASS`。由于 provider 字段遗漏，该作业实际训练的 tiny GDN 使用默认 16/32 heads 和 128 维。它只证明 Bridge→MCore→TE 的多卡执行、distributed Adam、DCP save 和跨进程 resume 路径，不证明声明的 tiny GDN 几何、QwenAir 结构或对应数值正确。
- MCore 定向作业 `4809119` 以 `COMPLETED 0:0` 结束，16/16 测试通过。该作业使用 MCore `830089a60` 加与 `70e662aac` 完全相同的分桶 patch，验证 core DDP、modular wrapper 与 legacy resolver 在同步模式下尊重显式有限 `bucket_size`/`num_buckets`，同时保持未显式配置时的单 bucket 行为。
- 最终 preflight 作业 `4810004` 固定 Bridge `431f2f175`、MCore `70e662aac`、TE `3250741d`，以 `COMPLETED 0:0` 结束。Bridge 测试 18/18、MCore 测试 13/13、同步 DDP bucket 测试 12/12 通过，并输出 `QWENAIR_BRIDGE_431_TRITON_PREFLIGHT_PASS`。检查覆盖修正后的五个 GDN `linear_*` 字段、显式 40M bucket 和最终依赖组合。
- 最终 8×B200 作业 `4810199` 在 `umbriel-b200-091` 上运行 8 分 41 秒，以 `COMPLETED 0:0` 结束；受测提交精确为 Bridge `431f2f17540af31a7fa40970f5a28be1bb2b7716`、MCore `70e662aac0ea4eb53fbf60b1c105f278ea41c8e4`、TE `3250741db1e06acd638da6124bc193405565f771`。运行配置显示 `bucket_size=40_000_000`、`log_interval=1`，tiny GDN 为 4/4 key/value heads、8/8 key/value dims，目标配置保持 48 value heads；router audit 观测到 227 组不同 expert 选择。iteration 1 的 LM loss 为 `4.857530`、gradient norm 为 `0.464`；第二个独立 `torchrun` 从 iteration 1 连同 `dp_reshardable` optimizer 恢复，iteration 2 的 LM loss 为 `4.856993`、gradient norm 为 `0.468`。两步 skipped/nan 均为 0；每代 checkpoint 含 8 个 rank shard、metadata 和 train state，tracker 最终为 2。作业输出 `QWENAIR_BRIDGE_431_TRITON_SAVE_RESUME_PASS`，stderr 无 `Traceback`。这构成修正 GDN 几何后的最终 8 卡 text base-LM、distributed Adam、DCP save/resume 证据。

### 5.1 最终提交验收与目标形状状态

- **最终精确提交 8×B200：通过。** 作业 `4810004` 和 `4810199` 覆盖最终 Bridge/MCore/TE 提交、修正后的 GDN 字段、显式 40M bucket、有限 loss/gradient、distributed optimizer、每 rank DCP shard 和独立新进程恢复；详细证据见上节。
- **目标文本几何 32×B300：等待资源。** 旧作业 `4806561` 固定在 MCore `a37c60dba`，且没有显式有限同步 DDP bucket；它已被取消，不计为通过。替代作业 `4809933` 固定最终功能提交，使用目标 48 层/512 experts/PLE、EP32、显式 `seq_length=8` 和一步训练；当前状态仍为 `PENDING (Priority)`，集群尚无 4 个完整可用 B300 节点，因此没有 loss、gradient、bucket 或峰值显存结果，不能称为通过。目标 recipe 的默认 64 token 仅用于较小 bring-up；4096 token 在当前 reference GDN 上仅保留 FP32 recurrence state 就约需 432 GiB/GPU。该门槛仍不包含 MTP、Vision、262K context、CP、PP、TP 或 ETP。

## 6. 当前支持边界与后续门槛

### 6.1 已支持的训练契约

1. **并行布局。** 当前仅支持 `TP=PP=CP=ETP=1`，且 `world_size=EP×EDP`、`num_experts % EP == 0`。EP 可以大于 1；TP、PP、CP 或 expert TP 大于 1 会在构造前明确报错。
2. **参数所有权。** 每个 EP rank 持有连续的 routed expert shard 和 PLE row shard。dense 参数属于 WORLD 副本，local expert/PLE shard 只在对应 EDP replica group 中归约。MCore DDP 与 distributed Adam 已替代早期测试循环中的手工复制参数梯度同步。
3. **loss 与 schedule。** Megatron GPTDataset 的 labels 被视为已右移；模型的 pipeline `set_input_tensor` 契约在 PP=1 下可用。CE 用全局有效 token 数归一化，router aux 保持跨 EP/EDP 的定义，避免 rank-local sequence length 不同造成缩放偏差。
4. **checkpoint 与重启。** model shard 和 distributed optimizer state 通过 MCore DCP 保存/恢复；PLE/expert 轴和 EDP replica ID 已编码到 sharded state。`--checkpoint-step 0` 可用于不写大 checkpoint 的 allocation smoke。
5. **初始化和 collective 大小。** expert 与 PLE 的随机初始化由逻辑参数身份决定，不随 EP shard 边界变化。同步 DDP 未显式配置 `bucket_size`/`num_buckets` 时仍使用单 bucket；调用方显式给出有限值时，MCore `70e662aac` 会尊重它。QwenAir 入口与 Bridge recipe 都显式使用 `bucket_size=40_000_000`。该值是累积参数的切桶目标，不是单参数硬上限，MCore 不会拆分单个参数。以目标 EP32 为例，local PLE 单 tensor 有约 `1,600,007,680` elements，虽低于 `2^31` collective 上限，其 FP32 gradient buffer 仍单独约 5.96 GiB。目标形状必须用实际 bucket report 和 peak-memory 验收。采用旧单-bucket layout 保存的 `dp_reshardable` optimizer checkpoint 可能无法直接加载到显式多 bucket layout；需要保留旧布局完成恢复，或先转成 `fully_reshardable` checkpoint 再迁移。model checkpoint 不受该 bucket layout 变化影响。
6. **QSA backend。** `dense` 与 `te_triton` 均已通过 MCore 多卡训练 gate；TE reference、indexed SDPA 和 Triton public API 均已实现。Bridge `te_triton` 的最终修正组合已由 `4810004` 和 `4810199` 完成 8×B200 preflight、训练、保存与独立进程恢复。

### 6.2 尚未闭合的 QwenAir 目标

1. **hard top-k indexer 训练。** 公开 HF forward 中离散 top-k 对 LM loss 没有 indexer gradient；资料只说明它使用独立目标，没有 teacher、label、loss、系数或 backward golden。当前 indexer 参数保留并预期 `main_grad=0`，不能把 base-LM gate 称为全部参数训练。
2. **MTP。** 目标 JSON 声明一层 MTP，但公开实现没有 next-token shift、loss 权重、detach 或参数共享规则。MTP 当前禁用，获得权威契约与 fixture 后再接入。
3. **Vision。** 视觉结构、输入 scatter 和位置 fixture 已有局部验证，尚未接入当前 text distributed trainer、optimizer ownership 和多卡 DCP；完整 image/video 训练未验收。
4. **262K 与 CP。** 当前 CP=1。GDN recurrent state/causal-conv halo、PLE ngram/EOS/conv halo、QSA global block ID、selector、远端 K/V、MRoPE 和 label 边界都没有 CP 协议。TE Triton prototype 也尚未在 262144 token 上完成容量和吞吐门槛。
5. **TP/PP/ETP。** embedding/LM head、HC、GDN、QSA、router/shared expert 仍未采用目标所需的 TP ownership；2 KV heads 与 1 indexer K head 需要复制及梯度规约方案。48 层尚未按 PP stage 持有，跨 stage router aux 和 PLE 固定全局层号也未验收。
6. **目标规模与效率。** `4805248` 只是 allocation-free 预算，没有构造或训练 176.944B 文本参数。当前 local expert 计算仍包含逐专家 Python 调度，PLE 与 expert 使用同一 sharding group；目标规模还需要 grouped GEMM、足够的节点、activation recompute、真实峰值显存与吞吐验证。
7. **真实 checkpoint 转换。** logical HF-compatible key、PLE/expert DCP shard 和合成 roundtrip 已有覆盖；真实 checkpoint manifest、128 文件 PLE 流式读取、跨拓扑 reshard 及 optimizer/data/RNG 完整恢复仍需真实权重验收。

### 6.3 下一阶段执行顺序

1. 将已经通过的 Bridge `431f2f175`/MCore `70e662aac`/TE `3250741d` 8×B200 gate 保持为回归门槛；等待作业 `4809933` 获得 4 个完整 B300 节点后，收集目标 EP32 `seq_length=8` 的 loss、gradient、bucket layout 与每 rank peak memory，再决定下一步内核和分片优先级。
2. 在当前支持网格内逐步扩大模型，记录每 rank peak memory 与 communication workspace，并验证逻辑 shard 初始化在 EP 变化后仍能复现；随后运行目标文本形状的短序列 allocation/one-step gate。
3. 分别实现 TP/ETP、PP 和 CP，不把标准 Transformer 的开关直接套用到 QwenAir。每一维先以 tiny oracle 对照输出、梯度、loss normalization 和跨拓扑 DCP，再组合多节点网格。
4. 将 GDN 分块 kernel、QSA selector/attention 262K 路径、grouped expert GEMM 和 PLE 存取逐项做数值、梯度、容量与 profiler 门槛；任何 math fallback 或二次临时量都必须显式记录。
5. 获得 indexer 独立目标、MTP 契约和真实 checkpoint 后补齐对应训练 loss、converter 与 golden；最后接入 Vision 并完成 image/video 多卡训练和 restart。

最终“QwenAir 可以正确训练”需要同时满足：目标文本/视觉结构、已定义的全部训练目标、目标上下文和目标并行网格均有有限梯度与可解释 loss；模型、optimizer、RNG 和 data position 可跨进程重启；数值与权威 oracle 在预定容差内一致。当前最终 Bridge+MCore+TE 组合已支持 EP×EDP 多卡 tiny text base-LM 训练和独立进程恢复；32×B300 目标文本形状尚在排队。MTP、Vision、262K、CP、PP、TP、ETP、真实 checkpoint 转换和完整目标形状训练仍未闭合，因此不支持完整原始预训练等价声明。
