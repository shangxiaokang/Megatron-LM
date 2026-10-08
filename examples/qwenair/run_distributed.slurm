#!/bin/bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Submit with site allocation flags, for example:
#   sbatch -A <account> -p <partition> -N 2 --gpus-per-node=8 run_distributed.slurm

#SBATCH --job-name=qwenair-train
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

: "${QWENAIR_WORKTREE:?set QWENAIR_WORKTREE to a shared Megatron-LM checkout}"
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29500}
TRAIN_ARGS=${TRAIN_ARGS:---expert-model-parallel-size 8 --steps 2 --checkpoint-step 1}

cd "${QWENAIR_WORKTREE}"
mkdir -p logs

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
NNODES=${SLURM_NNODES}
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
export MASTER_ADDR MASTER_PORT NNODES GPUS_PER_NODE WORLD_SIZE

srun --ntasks="${NNODES}" --ntasks-per-node=1 bash -c '
  NODE_RANK=${SLURM_NODEID}
  uv run python -m torch.distributed.run \
    --nnodes='"${NNODES}"' \
    --nproc-per-node='"${GPUS_PER_NODE}"' \
    --node-rank=${NODE_RANK} \
    --master-addr='"${MASTER_ADDR}"' \
    --master-port='"${MASTER_PORT}"' \
    examples/qwenair/train_distributed.py '"${TRAIN_ARGS}"'
'
