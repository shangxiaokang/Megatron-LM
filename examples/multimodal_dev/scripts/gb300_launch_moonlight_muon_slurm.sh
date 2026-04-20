#!/bin/bash
set -euxo pipefail

################################################################################
# Moonlight-16B Muon GB300/GB200 Slurm Launch Script
#
# - Reference style: megatron-moe-scripts/examples/qwen3/gb300_launch_235b_slurm.sh
# - Launch mode: Slurm + mpirun + python (NO torchrun)
################################################################################

#===============================================================================
# Cluster & Container Configuration
#===============================================================================
export ACCOUNT=${ACCOUNT:-general_sa}
export PARTITION=${PARTITION:-gb200-backfill}
export CLUSTER=${CLUSTER:-lyris}
export CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/general_sa/xshang/sqsh/Pytorch-2512-GB300-HybridEP-V2.sqsh}
export CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-/home/xshang:/home/xshang,/lustre/fsw/general_sa/:/lustre/fsw/general_sa/,/lustre/raplab/client/:/lustre/raplab/client/}

#===============================================================================
# Paths
#===============================================================================
export CODE_DIR=${CODE_DIR:-/home/xshang/Megatron-LM}
export WORKDIR=${WORKDIR:-/lustre/raplab/client/xshang/workspace/Megatron-LM/examples/multimodal_dev/scripts}
export TRAINING_SCRIPT_PATH=${TRAINING_SCRIPT_PATH:-${CODE_DIR}/pretrain_gpt.py}
export OUT_DIR=${OUT_DIR:-${WORKDIR}/output}
export LOG_DIR=${LOG_DIR:-${OUT_DIR}/slurm_logs}
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

#===============================================================================
# Runtime Configuration (overridable via env)
#===============================================================================
export NNODES=${NNODES:-1}
export SEGMENT=${SEGMENT:-${NNODES}}
export N_TASKS_PER_NODE=${N_TASKS_PER_NODE:-1}   # only one launcher task
export GPUS_PER_NODE=${GPUS_PER_NODE:-4}
export RUN_TIME=${RUN_TIME:-08:00:00}
export MASTER_PORT=${MASTER_PORT:-6000}
export JOB_NAME=${JOB_NAME:-Moonlight-16B-Muon}
export DRY_RUN=${DRY_RUN:-0}

#===============================================================================
# Environment variables for training
#===============================================================================
export NCCL_IB_SL=1
export NVTE_FUSED_ATTN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

#===============================================================================
# Training Parameters (from run_moonlight_muon.sh)
#===============================================================================
export TOKENIZER_MODEL=${TOKENIZER_MODEL:-moonshotai/Moonlight-16B-A3B-Instruct}
export DATA_PATH=${DATA_PATH:-/lustre/fsw/general_sa/xshang/dataset/imdb/imdb_megatron_text_document}
export CHECKPOINT=${CHECKPOINT:-/lustre/fsw/general_sa/xshang/checkpoints/Moonlight-16B}

export GBS=${GBS:-768}
export MBS=${MBS:-1}
export EP_SIZE=${EP_SIZE:-4}
export TP_SIZE=${TP_SIZE:-1}
export PP_SIZE=${PP_SIZE:-1}
export CP_SIZE=${CP_SIZE:-1}
export SEQ_LENGTH=${SEQ_LENGTH:-8192}
export PR=${PR:-bf16}

export WANDB_PROJECT=${WANDB_PROJECT:-Moonlight-16B}
export EXP_NAME=${EXP_NAME:-Moonlight-16B_tp${TP_SIZE}_ep${EP_SIZE}_pp${PP_SIZE}_${PR}_MBS${MBS}_GBS${GBS}_muon}

if [[ "${DRY_RUN}" -eq 0 ]]; then
    mkdir -p "${CHECKPOINT}"
fi

TRAINING_PARAMS=""

# Data args
TRAINING_PARAMS+=" --data-path ${DATA_PATH}"
TRAINING_PARAMS+=" --tokenizer-type HuggingFaceTokenizer"
TRAINING_PARAMS+=" --tokenizer-model ${TOKENIZER_MODEL}"
TRAINING_PARAMS+=" --trust-remote-code"
TRAINING_PARAMS+=" --tiktoken-pattern v2"
TRAINING_PARAMS+=" --split 90,5,5"
TRAINING_PARAMS+=" --num-workers 6"
TRAINING_PARAMS+=" --no-create-attention-mask-in-dataloader"

# Model parallel + MoE args
TRAINING_PARAMS+=" --distributed-timeout-minutes 120"
TRAINING_PARAMS+=" --micro-batch-size ${MBS}"
TRAINING_PARAMS+=" --global-batch-size ${GBS}"
TRAINING_PARAMS+=" --expert-model-parallel-size ${EP_SIZE}"
TRAINING_PARAMS+=" --tensor-model-parallel-size ${TP_SIZE}"
TRAINING_PARAMS+=" --pipeline-model-parallel-size ${PP_SIZE}"
TRAINING_PARAMS+=" --context-parallel-size ${CP_SIZE}"
TRAINING_PARAMS+=" --overlap-grad-reduce"
TRAINING_PARAMS+=" --num-experts 64"
TRAINING_PARAMS+=" --moe-layer-freq 1"
TRAINING_PARAMS+=" --moe-ffn-hidden-size 1408"
TRAINING_PARAMS+=" --moe-shared-expert-intermediate-size 2816"
TRAINING_PARAMS+=" --moe-router-load-balancing-type seq_aux_loss"
TRAINING_PARAMS+=" --moe-aux-loss-coeff 1e-3"
TRAINING_PARAMS+=" --moe-router-topk 6"
TRAINING_PARAMS+=" --moe-router-pre-softmax"
TRAINING_PARAMS+=" --moe-grouped-gemm"
TRAINING_PARAMS+=" --moe-router-dtype fp32"
TRAINING_PARAMS+=" --moe-router-topk-scaling-factor 2.446"
TRAINING_PARAMS+=" --moe-router-score-function sigmoid"
TRAINING_PARAMS+=" --moe-router-enable-expert-bias"
TRAINING_PARAMS+=" --moe-router-bias-update-rate 1e-3"
TRAINING_PARAMS+=" --moe-token-dispatcher-type alltoall"

# Model architecture
TRAINING_PARAMS+=" --untie-embeddings-and-output-weights"
TRAINING_PARAMS+=" --no-bias-swiglu-fusion"
TRAINING_PARAMS+=" --swiglu"
TRAINING_PARAMS+=" --use-mcore-models"
TRAINING_PARAMS+=" --transformer-impl transformer_engine"
TRAINING_PARAMS+=" --position-embedding-type rope"
TRAINING_PARAMS+=" --no-rope-fusion"
TRAINING_PARAMS+=" --rotary-percent 1.0"
TRAINING_PARAMS+=" --rotary-base 50000"
TRAINING_PARAMS+=" --normalization RMSNorm"
TRAINING_PARAMS+=" --norm-epsilon 1e-5"
TRAINING_PARAMS+=" --multi-latent-attention"
TRAINING_PARAMS+=" --num-attention-heads 16"
TRAINING_PARAMS+=" --attention-backend auto"
TRAINING_PARAMS+=" --hidden-dropout 0.0"
TRAINING_PARAMS+=" --attention-dropout 0.0"
TRAINING_PARAMS+=" --ckpt-format torch_dist"
TRAINING_PARAMS+=" --disable-bias-linear"
TRAINING_PARAMS+=" --kv-lora-rank 512"
TRAINING_PARAMS+=" --qk-pos-emb-head-dim 64"
TRAINING_PARAMS+=" --v-head-dim 128"
TRAINING_PARAMS+=" --seq-length ${SEQ_LENGTH}"
TRAINING_PARAMS+=" --num-layers 27"
TRAINING_PARAMS+=" --hidden-size 2048"
TRAINING_PARAMS+=" --qk-layernorm"
TRAINING_PARAMS+=" --max-position-embeddings ${SEQ_LENGTH}"
TRAINING_PARAMS+=" --cuda-graph-impl transformer_engine"
TRAINING_PARAMS+=" --cuda-graph-scope attn moe_router"

# Logging/checkpoint
TRAINING_PARAMS+=" --train-samples 133632768"
TRAINING_PARAMS+=" --log-interval 1"
TRAINING_PARAMS+=" --save-interval 100"
TRAINING_PARAMS+=" --eval-interval 1000"
TRAINING_PARAMS+=" --eval-iters 1"
TRAINING_PARAMS+=" --log-throughput"
TRAINING_PARAMS+=" --timing-log-level 0"
TRAINING_PARAMS+=" --tensorboard-log-interval 1"
TRAINING_PARAMS+=" --log-timers-to-tensorboard"
TRAINING_PARAMS+=" --log-memory-to-tensorboard"
TRAINING_PARAMS+=" --log-params-norm"
TRAINING_PARAMS+=" --log-num-zeros-in-grad"
TRAINING_PARAMS+=" --log-validation-ppl-to-tensorboard"
TRAINING_PARAMS+=" --tensorboard-dir ${OUT_DIR}"
TRAINING_PARAMS+=" --save ${CHECKPOINT}"
TRAINING_PARAMS+=" --wandb-project ${WANDB_PROJECT}"
TRAINING_PARAMS+=" --wandb-exp-name ${EXP_NAME}"
TRAINING_PARAMS+=" --wandb-save-dir ${CHECKPOINT}"

# Optimizer
TRAINING_PARAMS+=" --dist-ckpt-strictness raise_unexpected"
TRAINING_PARAMS+=" --use-distributed-optimizer"
TRAINING_PARAMS+=" --main-grads-dtype fp32"
TRAINING_PARAMS+=" --main-params-dtype fp32"
TRAINING_PARAMS+=" --cross-entropy-loss-fusion"
TRAINING_PARAMS+=" --cross-entropy-fusion-impl native"
TRAINING_PARAMS+=" --lr 0.00015"
TRAINING_PARAMS+=" --weight-decay 0.1"
TRAINING_PARAMS+=" --adam-beta1 0.9"
TRAINING_PARAMS+=" --adam-beta2 0.95"
TRAINING_PARAMS+=" --optimizer muon --muon-momentum 0.95 --muon-scale-mode spectral --muon-extra-scale-factor 0.2 --muon-no-split-qkv"
TRAINING_PARAMS+=" --clip-grad 1.0"
TRAINING_PARAMS+=" --lr-decay-style WSD"
TRAINING_PARAMS+=" --lr-wsd-decay-style linear"
TRAINING_PARAMS+=" --lr-wsd-decay-samples 5377"
TRAINING_PARAMS+=" --lr-warmup-samples 4000"
TRAINING_PARAMS+=" --min-lr 1.0e-5"
TRAINING_PARAMS+=" --manual-gc"
TRAINING_PARAMS+=" --manual-gc-interval 10"

# Precision args
TRAINING_PARAMS+=" --bf16"
if [[ ${PR} == "mxfp8" ]]; then
    TRAINING_PARAMS+=" --fp8-recipe mxfp8"
    TRAINING_PARAMS+=" --fp8-format e4m3"
    TRAINING_PARAMS+=" --moe-router-padding-for-quantization"
    TRAINING_PARAMS+=" --overlap-grad-reduce"
fi

# Auto resume when checkpoint exists
if [[ -f "${CHECKPOINT}/latest_checkpointed_iteration.txt" ]]; then
    TRAINING_PARAMS+=" --load ${CHECKPOINT}"
fi

# Final command (NO torchrun)
TRAINING_CMD="python ${TRAINING_SCRIPT_PATH} ${TRAINING_PARAMS} --distributed-backend nccl --auto-detect-ckpt-format"

#===============================================================================
# Submit
#===============================================================================
TIMESTAMP=$(date +'%y%m%d_%H%M%S')
SBATCH_ARG="--segment=${SEGMENT}"
SLURM_LOGS="${LOG_DIR}"

set +e
if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "============================================================"
    echo "=== DRY RUN - SLURM Job Script ==="
    echo "============================================================"
    cat <<EOF
#!/bin/bash

#SBATCH --nodes=${NNODES}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --ntasks-per-node=${N_TASKS_PER_NODE}
#SBATCH --time=${RUN_TIME}
#SBATCH --job-name=${ACCOUNT}-${JOB_NAME}-${TIMESTAMP}
#SBATCH --output=${SLURM_LOGS}/slurm-%j.log
#SBATCH --exclusive

set -euo pipefail

export MASTER_ADDR=\$(scontrol show hostnames "\${SLURM_JOB_NODELIST}" | head -n 1)
export WORLD_SIZE=\$((\${SLURM_NNODES} * ${GPUS_PER_NODE}))
echo "MASTER_ADDR=\${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}, WORLD_SIZE=\${WORLD_SIZE}"

srun \\
    --mpi=pmix -l \\
    --ntasks=1 \\
    --kill-on-bad-exit=1 \\
    --no-container-mount-home \\
    --container-image=${CONTAINER_IMAGE} \\
    --container-mounts=${CONTAINER_MOUNTS} \\
    --container-workdir=${WORKDIR} \\
    bash -lc "set -euo pipefail; \\
        pip install flask flask-restful uvloop; \\
        mpirun --allow-run-as-root --bind-to none --map-by ppr:${GPUS_PER_NODE}:node -np \\\${WORLD_SIZE} \\
            -x MASTER_ADDR -x MASTER_PORT -x WORLD_SIZE -x NCCL_IB_SL -x NVTE_FUSED_ATTN -x PYTORCH_CUDA_ALLOC_CONF -x CUDA_DEVICE_MAX_CONNECTIONS \\
            bash -lc 'set -euo pipefail; export RANK=\\\${OMPI_COMM_WORLD_RANK}; export LOCAL_RANK=\\\${OMPI_COMM_WORLD_LOCAL_RANK}; export WORLD_SIZE=\\\${OMPI_COMM_WORLD_SIZE}; export CUDA_VISIBLE_DEVICES=\\\${LOCAL_RANK}; ${TRAINING_CMD}'" \\
        2>&1 | tee ${SLURM_LOGS}/\\\${SLURM_JOB_ID}.log
EOF
    echo "============================================================"
    echo "=== Configuration Summary ==="
    echo "============================================================"
    echo "CLUSTER:          ${CLUSTER}"
    echo "PARTITION:        ${PARTITION}"
    echo "NODES:            ${NNODES}"
    echo "GPUS_PER_NODE:    ${GPUS_PER_NODE}"
    echo "WORLD_SIZE:       $((NNODES * GPUS_PER_NODE))"
    echo "RUN_TIME:         ${RUN_TIME}"
    echo "PRECISION(PR):    ${PR}"
    echo "CHECKPOINT:       ${CHECKPOINT}"
    echo "============================================================"
    echo "=== Training Command ==="
    echo "============================================================"
    echo "${TRAINING_CMD}"
    echo "============================================================"
else
    sbatch ${SBATCH_ARG} <<EOF
#!/bin/bash

#SBATCH --nodes=${NNODES}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --ntasks-per-node=${N_TASKS_PER_NODE}
#SBATCH --time=${RUN_TIME}
#SBATCH --job-name=${ACCOUNT}-${JOB_NAME}-${TIMESTAMP}
#SBATCH --output=${SLURM_LOGS}/slurm-%j.log
#SBATCH --exclusive

set -euo pipefail

export MASTER_ADDR=\$(scontrol show hostnames "\${SLURM_JOB_NODELIST}" | head -n 1)
export WORLD_SIZE=\$((\${SLURM_NNODES} * ${GPUS_PER_NODE}))
echo "MASTER_ADDR=\${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}, WORLD_SIZE=\${WORLD_SIZE}"

srun \\
    --mpi=pmix -l \\
    --ntasks=1 \\
    --kill-on-bad-exit=1 \\
    --no-container-mount-home \\
    --container-image=${CONTAINER_IMAGE} \\
    --container-mounts=${CONTAINER_MOUNTS} \\
    --container-workdir=${WORKDIR} \\
    bash -lc "set -euo pipefail; \\
        pip install flask flask-restful uvloop; \\
        mpirun --allow-run-as-root --bind-to none --map-by ppr:${GPUS_PER_NODE}:node -np \\\${WORLD_SIZE} \\
            -x MASTER_ADDR -x MASTER_PORT -x WORLD_SIZE -x NCCL_IB_SL -x NVTE_FUSED_ATTN -x PYTORCH_CUDA_ALLOC_CONF -x CUDA_DEVICE_MAX_CONNECTIONS \\
            bash -lc 'set -euo pipefail; export RANK=\\\${OMPI_COMM_WORLD_RANK}; export LOCAL_RANK=\\\${OMPI_COMM_WORLD_LOCAL_RANK}; export WORLD_SIZE=\\\${OMPI_COMM_WORLD_SIZE}; export CUDA_VISIBLE_DEVICES=\\\${LOCAL_RANK}; ${TRAINING_CMD}'" \\
        2>&1 | tee ${SLURM_LOGS}/\\\${SLURM_JOB_ID}.log
EOF
    echo "Job submitted successfully!"
fi
set -e
