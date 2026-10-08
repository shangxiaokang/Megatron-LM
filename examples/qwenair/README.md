# QwenAir distributed training gate

`train_distributed.py` is the multi-GPU training entry point for the staged
QwenAir text implementation. It uses explicit `ProcessGroupCollection`
instances throughout the model, MCore DDP, the MCore distributed Adam
optimizer, and MCore distributed checkpointing.

The supported layout is `TP=PP=CP=ETP=1`, with `world_size = EP * EDP`.
Dense parameters reduce across WORLD. Routed-expert weights and PLE row shards
reduce only across the EDP ranks holding the same local shard. The entry point
sets `calculate_per_token_loss=True` and weights each EP-local CE contribution
by its valid-token count over the WORLD token count. This remains a true global
token mean when EDP replicas have different sequence lengths. Router auxiliary
loss is averaged across EDP.

Run the small end-to-end gate on four GPUs:

```bash
uv run python -m torch.distributed.run --nproc-per-node 4 \
  examples/qwenair/train_distributed.py \
  --expert-model-parallel-size 2 \
  --steps 2 \
  --checkpoint-step 1 \
  --verify-restart \
  --checkpoint-dir /shared/checkpoints/qwenair-tiny \
  --output-json /shared/logs/qwenair-tiny.json
```

Validate a frozen target configuration without allocating the model:

```bash
PYTHONPATH=. uv run python examples/qwenair/train_distributed.py \
  --dry-run \
  --config-json /shared/configs/bf16-model-config.json \
  --world-size 512 \
  --expert-model-parallel-size 512 \
  --max-single-rank-ple-elements 110000000 \
  --max-single-rank-parameters 250000000
```

The dry run validates WORLD/EP/EDP and PLE/expert divisibility, prints every EP
and EDP rank group, and reports per-rank BF16 parameter, FP32 gradient, and
distributed Adam master/moment storage. It excludes activations and temporary
workspaces, so the result is a lower bound rather than a capacity guarantee.
Use `--checkpoint-step 0 --steps 1` for a target-shape allocation/training
smoke test where writing a very large checkpoint is undesirable.

For multi-node Slurm, submit `run_distributed.slurm` from a shared checkout and
pass site-specific allocation options to `sbatch`. The script uses one `srun`
task per node and one `torch.distributed.run` worker per visible GPU.

The hard top-k QSA indexer remains frozen by LM loss until an authoritative
indexer target/loss is available. MTP training remains disabled for the same
reason. Those boundaries do not prevent continued LM training with the
implemented text path, but they remain required for claiming exact original
pretraining reproduction.
