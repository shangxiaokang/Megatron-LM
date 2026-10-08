# QwenAir distributed training gate

`train_distributed.py` is the multi-GPU entry point for the staged QwenAir text implementation. The current functional code point is MCore `70e662aac0ea4eb53fbf60b1c105f278ea41c8e4`; Bridge `431f2f17540af31a7fa40970f5a28be1bb2b7716` pins that commit, and both dependency locks pin TransformerEngine `3250741db1e06acd638da6124bc193405565f771`. The entry point uses explicit `ProcessGroupCollection` instances, MCore DDP, distributed Adam and MCore distributed checkpointing.

The supported layout is `TP=PP=CP=ETP=1`, with `world_size = EP * EDP` and `num_experts % EP == 0`. EP greater than one is supported. Dense parameters reduce across WORLD; routed-expert weights and PLE row shards reduce only across the EDP ranks that hold the same local shard. The model initializes expert and PLE shards from their logical parameter identity, so values do not depend on runtime EP shard boundaries.

The entry point sets `calculate_per_token_loss=True` and weights each EP-local CE contribution by its valid-token count over the WORLD token count. This gives a global token mean when EDP replicas have different sequence lengths. Router auxiliary loss follows the EDP reduction contract. Megatron GPTDataset labels are already shifted and are not shifted a second time.

Run the tiny end-to-end dense QSA gate on four GPUs:

```bash
uv run python -m torch.distributed.run --nproc-per-node 4 \
  examples/qwenair/train_distributed.py \
  --expert-model-parallel-size 2 \
  --qsa-backend dense \
  --steps 2 \
  --checkpoint-step 1 \
  --verify-restart \
  --checkpoint-dir /shared/checkpoints/qwenair-tiny \
  --output-json /shared/logs/qwenair-tiny.json
```

This exercises BF16 forward/backward, dense and expert DDP buckets, distributed Adam, model/optimizer DCP, reload and exact replay. `--checkpoint-step 0 --steps 1` disables checkpoint writing for an allocation smoke test.

Validate the frozen target configuration without allocating its parameters:

```bash
PYTHONPATH=. uv run python examples/qwenair/train_distributed.py \
  --dry-run \
  --config-json /shared/configs/bf16-model-config.json \
  --world-size 512 \
  --expert-model-parallel-size 512 \
  --max-single-rank-ple-elements 110000000 \
  --max-single-rank-parameters 250000000
```

The dry run validates WORLD/EP/EDP and PLE/expert divisibility, prints each EP and EDP rank group, and reports per-rank BF16 parameter, FP32 gradient and distributed Adam storage. Job `4805248` validated `world=EP=512`, `EDP=1` for the 176,943,899,520-parameter text config and reported a 33.387 GiB/rank storage lower bound: 9.842 GiB BF16 parameters, 19.683 GiB FP32 gradients and 3.862 GiB distributed Adam master/m/v. It excludes activations, communication workspaces, allocator fragmentation and checkpoint staging, so it is not a capacity guarantee.

For multi-node Slurm, submit `run_distributed.slurm` from a shared checkout and pass site allocation options to `sbatch`. The script uses one `srun` task per node and one `torch.distributed.run` worker per visible GPU.

## Verified evidence

- 4×B300 job `4804756` and 8×B200 job `4805178` passed the dense QSA base-LM training, distributed optimizer and DCP/restart gate.
- 4×B300 job `4805736` passed topology-independent expert/PLE initialization and the tiny-model dense/expert bucket report.
- TransformerEngine `3250741d` job `4805785` passed compatible QSA public APIs plus BF16 forward and dQ/dK/dV backward at the target `24:2` GQA geometry.
- TE job `4804303` completed 201/201 QSA correctness tests and the Triton prototype benchmark. This is TE kernel evidence; it is not a passed MCore or Bridge multi-GPU `te_triton` training combination.
- 4×B200 job `4806499` tested MCore `a37c60dba` with TE `3250741d` and completed with exit code 0. Initialization tests passed 5/5; the router audit covered all four layers with 64 real tokens per layer and observed six distinct top-k expert selections. Distributed tests passed 7/7 and focused TE tests passed 6/6. The EP2×EDP2 `te_triton` run completed two distributed-Adam steps (loss `4.183996 → 4.164544`, gradient norm `0.649485 → 0.594810`) and exactly replayed model plus optimizer from four DCP shards.
- Preflight job `4808610`, fixed at Bridge `622681085`, MCore `830089a60` and TE `3250741d`, completed with exit code 0. It validated the real Bridge training import, lazy recipes, strict NullTokenizer asset save, the B200 native TE overlay and exact QSA Python source. Bridge tests passed 18/18, MCore tests passed 13/13, the router was nonzero with 225 distinct expert selections, and the job emitted `QWENAIR_BRIDGE_TRITON_PREFLIGHT_PASS`. Those tests did not assert the five GDN `linear_*` fields, so this is import/API/path evidence rather than evidence for correct GDN geometry.
- 8×B200 job `4808736` used the same exact commits and completed EP4×EDP2 BF16 `te_triton` training with distributed Adam and exit code 0. Iteration 1 had LM loss `4.852964878`, router aux `2.000883818` and gradient norm `0.415851295`. A clean second process found `iter_0000001`, restored model plus the `dp_reshardable` optimizer, and completed iteration 2 with LM loss `4.851634979`, router aux `2.000961304` and gradient norm `0.4181646705`; it emitted `QWENAIR_BRIDGE_TRITON_SAVE_RESUME_PASS`. The provider omitted five `linear_*` fields, so this run actually used the `TransformerConfig` defaults of 16/32 key/value heads and 128-dimensional GDN projections instead of the tiny fixture geometry. It proves distributed execution, optimizer, save and resume paths for the instantiated model; it does not prove the declared tiny GDN structure or QwenAir numerical correctness.
- MCore bucket job `4809119` completed 16/16 tests with exit code 0. It ran MCore `830089a60` plus the exact patch committed as `70e662aac`, verifying synchronous DDP behavior in core DDP, the modular wrapper and the legacy resolver.
- Final preflight job `4810004`, fixed at Bridge `431f2f175`, MCore `70e662aac` and TE `3250741d`, completed with exit code 0. Bridge tests passed 18/18, MCore tests passed 13/13 and synchronous DDP bucket tests passed 12/12. It checked the corrected GDN `linear_*` fields and emitted `QWENAIR_BRIDGE_431_TRITON_PREFLIGHT_PASS`.
- Final 8×B200 job `4810199` ran on `umbriel-b200-091` for 8 minutes 41 seconds and completed with exit code 0. It used the exact full commits listed at the top, `bucket_size=40_000_000` and `log_interval=1`. The instantiated tiny GDN retained 4/4 key/value heads and 8/8 key/value dimensions, while the target config retained 48 value heads; the router audit found 227 distinct expert selections. Iteration 1 had LM loss `4.857530` and gradient norm `0.464`. A second independent `torchrun` restored iteration 1 and its `dp_reshardable` optimizer, then produced iteration 2 LM loss `4.856993` and gradient norm `0.468`. Both iterations reported zero skipped iterations and zero NaNs. Each generation contained eight rank shards plus metadata and train state, the tracker finished at 2, stderr contained no `Traceback`, and the job emitted `QWENAIR_BRIDGE_431_TRITON_SAVE_RESUME_PASS`.

## Final-commit validation record

The final exact combination is Bridge `431f2f17540af31a7fa40970f5a28be1bb2b7716`, MCore `70e662aac0ea4eb53fbf60b1c105f278ea41c8e4` and TE `3250741db1e06acd638da6124bc193405565f771`. Bridge forwards `linear_conv_kernel_dim`, `linear_key_head_dim`, `linear_value_head_dim`, `linear_num_key_heads` and `linear_num_value_heads` from the QwenAir config. Jobs `4810004` and `4810199` validate this exact combination, the corrected tiny and target GDN fields, explicit DDP buckets, finite training metrics, eight-shard checkpoints and independent-process optimizer resume.

The target-text recipe now defaults to sequence length 64 rather than 4096. The reference GDN recurrence retains an FP32 state for each token during backward; across the 36 target GDN layers, 4096 tokens have a lower bound of about 432 GiB/GPU for that state alone. Sequence length 64 is a short-context bring-up setting, not a long-context validation. The replacement 32×B300 run will explicitly use sequence length 8.

The old target-text 32×B300 job `4806561` used MCore `a37c60dba` without the finite synchronous bucket fix and was canceled. Replacement job `4809933` uses the final functional commits and requests the target 48-layer/512-expert/PLE EP32 layout, correct 16/48-head GDN geometry and explicit sequence length 8. It remains `PENDING (Priority)` because four complete B300 nodes are not currently available. It has produced no loss, gradient, bucket or peak-memory result and is not a passed target-shape record. Even if it passes, this gate excludes MTP, Vision, 262K context, CP, PP, TP and ETP.

## Scope

`dense` and `te_triton` are proven by MCore multi-GPU training jobs. `te_reference` and `te_indexed_sdpa` remain validation paths. Final Bridge `431f2f175` with corrected GDN provider fields has passed the 8×B200 `te_triton` training, save and independent-process resume gate.

Synchronous DDP keeps its historical one-bucket behavior when neither `bucket_size` nor `num_buckets` is specified. MCore `70e662aac` honors an explicitly configured finite value, and the QwenAir entry point and Bridge recipe both set `bucket_size=40_000_000`. This value is a grouping target, not a hard parameter limit: MCore does not split one parameter across buckets. At EP32, one local target PLE tensor contains about `1,600,007,680` elements. It stays below the `2^31` collective limit, but its FP32 gradient buffer alone is about 5.96 GiB. A passing tiny-model bucket test does not establish target-shape memory capacity or communication efficiency.

An old `dp_reshardable` optimizer checkpoint written with a single-bucket layout may not load directly after changing to the explicit multi-bucket layout. Resume it under the old layout or migrate through a `fully_reshardable` checkpoint. Model checkpoints are independent of this optimizer bucket-layout change.

The hard top-k QSA indexer has no LM gradient until its authoritative independent target/loss is available. MTP training is disabled because its shift, loss and weight-sharing contract is unpublished. Vision is not part of this text trainer. TP, PP, CP and expert TP greater than one remain fail-closed, and the 262K CP/state/halo/remote-KV protocol is not implemented. The final Bridge+MCore+TE combination has verified the EP×EDP multi-GPU tiny text base-LM path and independent-process restore; the 32×B300 target-text shape is still waiting for resources. MTP, Vision, 262K, CP, PP, TP, ETP, real-checkpoint conversion and full target-shape training are not complete, so these results do not establish complete original QwenAir pretraining reproduction.
