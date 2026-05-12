#!/bin/bash
set -euxo pipefail

################################################################################
# Qwen3.5-VL Muon GB300 Slurm Launch Script
#
# - Reference style: gb300_launch_moonlight_muon_slurm.sh
# - Source params: run_qwen35_vl_muon.sh
# - Launch mode: Slurm + srun(pmix) + python (NO torchrun)
################################################################################

#===============================================================================
# Cluster & Container Configuration
#===============================================================================
export ACCOUNT=${ACCOUNT:-general_sa}
export PARTITION=${PARTITION:-gb300-backfill}
export CLUSTER=${CLUSTER:-lyris}
export CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/general_sa/xshang/sqsh/Pytorch-2512-GB300-HybridEP-Qwen3.5-TE2D.sqsh}
export CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-/home/xshang:/home/xshang,/lustre/fsw/general_sa/:/lustre/fsw/general_sa/}

#===============================================================================
# Paths
#===============================================================================
export CODE_DIR=${CODE_DIR:-/home/xshang/Megatron-LM}
export WORKDIR=${WORKDIR:-/home/xshang/Megatron-LM/examples/multimodal_dev/scripts}
export TRAINING_SCRIPT_PATH=${TRAINING_SCRIPT_PATH:-${CODE_DIR}/examples/multimodal_dev/pretrain_multimodal.py}
export OUT_DIR=${OUT_DIR:-${WORKDIR}/output}
export LOG_DIR=${LOG_DIR:-${OUT_DIR}/slurm_logs}
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

#===============================================================================
# Runtime Configuration (overridable via env)
#===============================================================================
export NNODES=${NNODES:-2}
export SEGMENT=${SEGMENT:-${NNODES}}
export GPUS_PER_NODE=${GPUS_PER_NODE:-4}
export N_TASKS_PER_NODE=${N_TASKS_PER_NODE:-${GPUS_PER_NODE}}
export RUN_TIME=${RUN_TIME:-08:00:00}
export MASTER_PORT=${MASTER_PORT:-6000}
export JOB_NAME=${JOB_NAME:-Qwen35-VL-Muon}
export DRY_RUN=${DRY_RUN:-0}
export MAX_RESTARTS=${MAX_RESTARTS:-100}
export REQUEUE_SIGNAL_SECONDS=${REQUEUE_SIGNAL_SECONDS:-300}

#===============================================================================
# Environment variables for training
#===============================================================================
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NCCL_IB_SL=1
export NCCL_GRAPH_REGISTER=0
export NVTE_FUSED_ATTN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=${NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN:-16}

#===============================================================================
# Qwen3.5-VL configuration (from run_qwen35_vl_muon.sh)
#===============================================================================
export MODEL_VARIANT=${MODEL_VARIANT:-35b_a3b}
export VISION_NUM_LAYERS=${VISION_NUM_LAYERS:-}
export PR=${PR:-mxfp8}
export DATASET_PROVIDER=${DATASET_PROVIDER:-text}

MXFP8=${MXFP8:-2D}
if [[${PR} == 'mxfp8' && ${MXFP8} == "2D" ]]; then
  export NVTE_MXFP8_ENABLE_2D_QUANTIZATION=1
fi


export MBS=${MBS:-4}
export GBS=${GBS:-1024}
export TP=${TP:-1}
export EP=${EP:-8}
export PP=${PP:-1}
export CP=${CP:-1}
export SEQ_LEN=${SEQ_LEN:-4096}
export TRAIN_SAMPLES=${TRAIN_SAMPLES:-268554687}
export LR_WARMUP_SAMPLES=${LR_WARMUP_SAMPLES:-$((100 * GBS))}
export LR_DECAY_SAMPLES=${LR_DECAY_SAMPLES:-${TRAIN_SAMPLES}}

case "${MODEL_VARIANT}" in
    proxy)
        NUM_LAYERS=${NUM_LAYERS:-4}
        NUM_EXPERTS=${NUM_EXPERTS:-16}
        HIDDEN_SIZE=4096
        FFN_HIDDEN_SIZE=10240
        NUM_ATTN_HEADS=32
        NUM_QUERY_GROUPS=2
        LINEAR_NUM_VALUE_HEADS=64
        VISION_NUM_LAYERS=${VISION_NUM_LAYERS:-2}
        ;;
    9b)
        NUM_LAYERS=${NUM_LAYERS:-32}
        NUM_EXPERTS=${NUM_EXPERTS:-0}
        HIDDEN_SIZE=4096
        FFN_HIDDEN_SIZE=12288
        NUM_ATTN_HEADS=16
        NUM_QUERY_GROUPS=4
        LINEAR_NUM_VALUE_HEADS=32
        VISION_NUM_LAYERS=${VISION_NUM_LAYERS:-27}
        ;;
    35b_a3b)
        NUM_LAYERS=${NUM_LAYERS:-40}
        NUM_EXPERTS=${NUM_EXPERTS:-256}
        HIDDEN_SIZE=2048
        FFN_HIDDEN_SIZE=4096
        NUM_ATTN_HEADS=16
        NUM_QUERY_GROUPS=2
        LINEAR_NUM_VALUE_HEADS=32
        VISION_NUM_LAYERS=${VISION_NUM_LAYERS:-27}
        ;;
    35b_a3b_light)
        NUM_LAYERS=${NUM_LAYERS:-12}
        NUM_EXPERTS=${NUM_EXPERTS:-128}
        HIDDEN_SIZE=2048
        FFN_HIDDEN_SIZE=4096
        NUM_ATTN_HEADS=16
        NUM_QUERY_GROUPS=2
        LINEAR_NUM_VALUE_HEADS=32
        VISION_NUM_LAYERS=${VISION_NUM_LAYERS:-7}
        ;;
    397b_a17b)
        NUM_LAYERS=${NUM_LAYERS:-60}
        NUM_EXPERTS=${NUM_EXPERTS:-512}
        HIDDEN_SIZE=4096
        FFN_HIDDEN_SIZE=10240
        NUM_ATTN_HEADS=32
        NUM_QUERY_GROUPS=2
        LINEAR_NUM_VALUE_HEADS=64
        VISION_NUM_LAYERS=${VISION_NUM_LAYERS:-27}
        ;;
    *)
        : "${NUM_LAYERS:?NUM_LAYERS must be set for MODEL_VARIANT=${MODEL_VARIANT}}"
        : "${NUM_EXPERTS:?NUM_EXPERTS must be set for MODEL_VARIANT=${MODEL_VARIANT}}"
        : "${HIDDEN_SIZE:?HIDDEN_SIZE must be set for MODEL_VARIANT=${MODEL_VARIANT}}"
        : "${FFN_HIDDEN_SIZE:?FFN_HIDDEN_SIZE must be set for MODEL_VARIANT=${MODEL_VARIANT}}"
        : "${NUM_ATTN_HEADS:?NUM_ATTN_HEADS must be set for MODEL_VARIANT=${MODEL_VARIANT}}"
        : "${NUM_QUERY_GROUPS:?NUM_QUERY_GROUPS must be set for MODEL_VARIANT=${MODEL_VARIANT}}"
        : "${LINEAR_NUM_VALUE_HEADS:?LINEAR_NUM_VALUE_HEADS must be set for MODEL_VARIANT=${MODEL_VARIANT}}"
        VISION_NUM_LAYERS=${VISION_NUM_LAYERS:-27}
        ;;
esac

export WANDB_PROJECT=${WANDB_PROJECT:-multimodal-v2-qwen35-vl}
export EXP_NAME=${EXP_NAME:-qwen35vl_${MODEL_VARIANT}_tp${TP}_ep${EP}_pp${PP}_${PR}_MBS${MBS}_GBS${GBS}_MUON_Silu}

export RECOMPUTE_VISION=${RECOMPUTE_VISION:-0}
if [[ "${RECOMPUTE_VISION}" -eq 1 ]]; then
    EXP_NAME+="_recompute_encoder"
fi
export RECOMPUTE=${RECOMPUTE:-0}
if [[ "${RECOMPUTE}" -eq 1 ]]; then
    EXP_NAME+="_recompute_decoder"
fi

export ROOT_DIR=${ROOT_DIR:-/lustre/fsw/general_sa/xshang/Qwen3.5}
export CHECKPOINT_STORE_PATH=${CHECKPOINT_STORE_PATH:-${ROOT_DIR}/${PR}-muon}
export TENSORBOARD_LOGS_PATH=${TENSORBOARD_LOGS_PATH:-${ROOT_DIR}/logs}
export DATA_PATH=${DATA_PATH:-/lustre/fsw/general_sa/xshang/dataset/peS2o/data/v2/pes2o_merged_v2_qwen3_5_text_document}
export SPLIT=${SPLIT:-969,30,1}

if [[ "${DRY_RUN}" -eq 0 ]]; then
    mkdir -p "${CHECKPOINT_STORE_PATH}" "${TENSORBOARD_LOGS_PATH}"
fi

#===============================================================================
# Build Training Command
#===============================================================================
TRAINING_PARAMS=""

# Core training + optimizer args
TRAINING_PARAMS+=" --micro-batch-size ${MBS}"
TRAINING_PARAMS+=" --global-batch-size ${GBS}"
TRAINING_PARAMS+=" --train-samples ${TRAIN_SAMPLES}"
TRAINING_PARAMS+=" --adam-beta1 0.9 --adam-beta2 0.95"
TRAINING_PARAMS+=" --optimizer muon --muon-momentum 0.95 --muon-scale-mode spectral --muon-extra-scale-factor 0.2 --muon-no-split-qkv"
TRAINING_PARAMS+=" --lr 1.2e-4"
TRAINING_PARAMS+=" --min-lr 1.2e-5"
TRAINING_PARAMS+=" --lr-decay-style cosine"
TRAINING_PARAMS+=" --lr-warmup-samples ${LR_WARMUP_SAMPLES}"
TRAINING_PARAMS+=" --lr-decay-samples ${LR_DECAY_SAMPLES}"
TRAINING_PARAMS+=" --weight-decay 0.1"
TRAINING_PARAMS+=" --clip-grad 1.0"
TRAINING_PARAMS+=" --bf16"
TRAINING_PARAMS+=" --use-mcore-models"
TRAINING_PARAMS+=" --use-flash-attn"
TRAINING_PARAMS+=" --transformer-impl transformer_engine"
TRAINING_PARAMS+=" --cross-entropy-loss-fusion"
TRAINING_PARAMS+=" --cross-entropy-fusion-impl te"
TRAINING_PARAMS+=" --enable-experimental"
TRAINING_PARAMS+=" --manual-gc"
TRAINING_PARAMS+=" --manual-gc-interval 5"
TRAINING_PARAMS+=" --mtp-num-layers 1"
TRAINING_PARAMS+=" --mtp-loss-scaling-factor 0.1"
TRAINING_PARAMS+=" --cuda-graph-impl transformer_engine"
TRAINING_PARAMS+=" --cuda-graph-scope attn moe_router"

# Precision mode args
if [[ "${PR}" == "mxfp8" ]]; then
    TRAINING_PARAMS+=" --fp8-recipe mxfp8"
    TRAINING_PARAMS+=" --fp8-format e4m3"
    TRAINING_PARAMS+=" --overlap-grad-reduce"
    TRAINING_PARAMS+=" --moe-router-padding-for-quantization"
fi

# Parallel args
TRAINING_PARAMS+=" --tensor-model-parallel-size ${TP}"
TRAINING_PARAMS+=" --pipeline-model-parallel-size ${PP}"
TRAINING_PARAMS+=" --expert-model-parallel-size ${EP}"
TRAINING_PARAMS+=" --context-parallel-size ${CP}"
TRAINING_PARAMS+=" --expert-tensor-parallel-size 1"
TRAINING_PARAMS+=" --use-distributed-optimizer"
TRAINING_PARAMS+=" --sequence-parallel"

# Logging/checkpointing args
TRAINING_PARAMS+=" --log-interval 1"
TRAINING_PARAMS+=" --save-interval 100"
TRAINING_PARAMS+=" --save-retain-interval 1000"
TRAINING_PARAMS+=" --eval-interval 1000"
TRAINING_PARAMS+=" --save ${CHECKPOINT_STORE_PATH}"
TRAINING_PARAMS+=" --eval-iters 10"
TRAINING_PARAMS+=" --tensorboard-dir ${TENSORBOARD_LOGS_PATH}"
TRAINING_PARAMS+=" --wandb-project ${WANDB_PROJECT}"
TRAINING_PARAMS+=" --wandb-exp-name ${EXP_NAME}"
TRAINING_PARAMS+=" --wandb-save-dir ${CHECKPOINT_STORE_PATH}"
TRAINING_PARAMS+=" --log-throughput"

# Tokenizer args
TRAINING_PARAMS+=" --tokenizer-type HuggingFaceTokenizer"
TRAINING_PARAMS+=" --tokenizer-model Qwen/Qwen3.5-35B-A3B"
TRAINING_PARAMS+=" --vocab-size 248320"

# Multimodal dataset/model args
TRAINING_PARAMS+=" --model-arch qwen35_vl"
TRAINING_PARAMS+=" --model-variant ${MODEL_VARIANT}"
TRAINING_PARAMS+=" --dataset-provider ${DATASET_PROVIDER}"
TRAINING_PARAMS+=" --data-path ${DATA_PATH}"
TRAINING_PARAMS+=" --split ${SPLIT}"
TRAINING_PARAMS+=" --image-token-id 248056"
TRAINING_PARAMS+=" --image-size 224"
TRAINING_PARAMS+=" --total-seq-length ${SEQ_LEN}"
TRAINING_PARAMS+=" --image-seq-length 256"
TRAINING_PARAMS+=" --vision-num-layers ${VISION_NUM_LAYERS}"

# Qwen3.5 decoder args
TRAINING_PARAMS+=" --num-layers ${NUM_LAYERS}"
TRAINING_PARAMS+=" --hidden-size ${HIDDEN_SIZE}"
TRAINING_PARAMS+=" --ffn-hidden-size ${FFN_HIDDEN_SIZE}"
TRAINING_PARAMS+=" --num-attention-heads ${NUM_ATTN_HEADS}"
TRAINING_PARAMS+=" --group-query-attention"
TRAINING_PARAMS+=" --num-query-groups ${NUM_QUERY_GROUPS}"
TRAINING_PARAMS+=" --kv-channels 256"
TRAINING_PARAMS+=" --max-position-embeddings 262144"
TRAINING_PARAMS+=" --seq-length ${SEQ_LEN}"
TRAINING_PARAMS+=" --normalization RMSNorm"
TRAINING_PARAMS+=" --apply-layernorm-1p"
TRAINING_PARAMS+=" --norm-epsilon 1e-06"
# TRAINING_PARAMS+=" --swiglu"
TRAINING_PARAMS+=" --disable-bias-linear"
TRAINING_PARAMS+=" --untie-embeddings-and-output-weights"
TRAINING_PARAMS+=" --position-embedding-type rope"
TRAINING_PARAMS+=" --rotary-percent 0.25"
TRAINING_PARAMS+=" --rotary-base 10000000"
TRAINING_PARAMS+=" --rotary-seq-len-interpolation-factor 1"
TRAINING_PARAMS+=" --qk-layernorm"
TRAINING_PARAMS+=" --attention-output-gate"
TRAINING_PARAMS+=" --attention-dropout 0.0"
TRAINING_PARAMS+=" --hidden-dropout 0.0"
TRAINING_PARAMS+=" --experimental-attention-variant gated_delta_net"
TRAINING_PARAMS+=" --linear-attention-freq 4"
TRAINING_PARAMS+=" --linear-conv-kernel-dim 4"
TRAINING_PARAMS+=" --linear-key-head-dim 128"
TRAINING_PARAMS+=" --linear-value-head-dim 128"
TRAINING_PARAMS+=" --linear-num-key-heads 16"
TRAINING_PARAMS+=" --linear-num-value-heads ${LINEAR_NUM_VALUE_HEADS}"
TRAINING_PARAMS+=" --make-vocab-size-divisible-by 485"

# MoE args
if [[ "${MODEL_VARIANT}" != "9b" ]]; then
    case "${MODEL_VARIANT}" in
        proxy)
            MOE_TOPK=2
            MOE_FFN_HIDDEN=1024
            MOE_SHARED_HIDDEN=1024
            ;;
        35b_a3b|35b_a3b_light)
            MOE_TOPK=8
            MOE_FFN_HIDDEN=512
            MOE_SHARED_HIDDEN=512
            ;;
        397b_a17b)
            MOE_TOPK=10
            MOE_FFN_HIDDEN=1024
            MOE_SHARED_HIDDEN=1024
            ;;
        *)
            MOE_TOPK=8
            MOE_FFN_HIDDEN=512
            MOE_SHARED_HIDDEN=512
            ;;
    esac
    TRAINING_PARAMS+=" --num-experts ${NUM_EXPERTS}"
    TRAINING_PARAMS+=" --moe-ffn-hidden-size ${MOE_FFN_HIDDEN}"
    TRAINING_PARAMS+=" --moe-shared-expert-intermediate-size ${MOE_SHARED_HIDDEN}"
    TRAINING_PARAMS+=" --moe-shared-expert-gate"
    TRAINING_PARAMS+=" --moe-router-load-balancing-type aux_loss"
    TRAINING_PARAMS+=" --moe-router-topk ${MOE_TOPK}"
    TRAINING_PARAMS+=" --moe-grouped-gemm"
    TRAINING_PARAMS+=" --moe-aux-loss-coeff 1e-3"
    TRAINING_PARAMS+=" --moe-token-dispatcher-type alltoall"
    TRAINING_PARAMS+=" --moe-router-dtype fp32"
fi

# Optional recompute args
if [[ "${RECOMPUTE}" -eq 1 ]]; then
    TRAINING_PARAMS+=" --recompute-granularity full"
    TRAINING_PARAMS+=" --recompute-method uniform"
    TRAINING_PARAMS+=" --recompute-num-layers 1"
fi
if [[ "${RECOMPUTE_VISION}" -eq 1 ]]; then
    TRAINING_PARAMS+=" --recompute-vision"
fi

# Optional FSDP args
export USE_FSDP=${USE_FSDP:-0}
if [[ "${USE_FSDP}" -eq 1 ]]; then
    TRAINING_PARAMS+=" --use-megatron-fsdp"
    TRAINING_PARAMS+=" --data-parallel-sharding-strategy optim_grads_params"
    TRAINING_PARAMS+=" --no-gradient-accumulation-fusion"
    TRAINING_PARAMS+=" --init-model-with-meta-device"
    TRAINING_PARAMS+=" --use-distributed-optimizer"
    TRAINING_PARAMS+=" --ckpt-format fsdp_dtensor"
    export CUDA_DEVICE_MAX_CONNECTIONS=8
fi

# Auto resume when checkpoint exists
if [[ -f "${CHECKPOINT_STORE_PATH}/latest_checkpointed_iteration.txt" ]]; then
    TRAINING_PARAMS+=" --load ${CHECKPOINT_STORE_PATH}"
fi

TRAINING_CMD="python ${TRAINING_SCRIPT_PATH} ${TRAINING_PARAMS}"

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
#SBATCH --job-name=${JOB_NAME}-${ACCOUNT}-${TIMESTAMP}
#SBATCH --output=${SLURM_LOGS}/slurm-%j.log
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --signal=B:USR1@${REQUEUE_SIGNAL_SECONDS}

set -euo pipefail

export MASTER_ADDR=\$(scontrol show hostnames "\${SLURM_JOB_NODELIST}" | head -n 1)
export WORLD_SIZE=\${SLURM_NTASKS}
echo "MASTER_ADDR=\${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}, WORLD_SIZE=\${WORLD_SIZE}"
echo "SLURM_RESTART_COUNT=\${SLURM_RESTART_COUNT:-0}, MAX_RESTARTS=${MAX_RESTARTS}"

if (( \${SLURM_RESTART_COUNT:-0} >= ${MAX_RESTARTS} )); then
    echo "[ERROR] Restart limit reached before launch: \${SLURM_RESTART_COUNT:-0}/${MAX_RESTARTS}"
    exit 1
fi

TIMEOUT_TRIGGERED=0
SRUN_PID=""
trap 'TIMEOUT_TRIGGERED=1; echo "[WARN] Received USR1 (time limit approaching), will requeue if allowed."; if [[ -n "\${SRUN_PID}" ]] && kill -0 "\${SRUN_PID}" 2>/dev/null; then kill -TERM "\${SRUN_PID}" || true; fi' USR1

set +e
srun \\
    --mpi=pmix -l \\
    --kill-on-bad-exit=1 \\
    --no-container-mount-home \\
    --container-image=${CONTAINER_IMAGE} \\
    --container-mounts=${CONTAINER_MOUNTS} \\
    --container-workdir=${WORKDIR} \\
    bash -lc "set -euo pipefail; \\
        ${TRAINING_CMD}" \\
        2>&1 | tee ${SLURM_LOGS}/\\\${SLURM_JOB_ID}.log &
SRUN_PID=\$!
wait "\${SRUN_PID}"
TRAIN_EXIT=\$?
set -e

if (( TIMEOUT_TRIGGERED == 1 )); then
    if (( \${SLURM_RESTART_COUNT:-0} < ${MAX_RESTARTS} )); then
        echo "[INFO] Requeue on timeout signal (\${SLURM_RESTART_COUNT:-0}/${MAX_RESTARTS})."
        scontrol requeue "\${SLURM_JOB_ID}" || true
        exit 0
    fi
    echo "[ERROR] Timeout signal received but max restarts reached."
    exit 1
fi

if (( TRAIN_EXIT == 0 )); then
    echo "[INFO] Training finished successfully."
    exit 0
fi

if (( \${SLURM_RESTART_COUNT:-0} < ${MAX_RESTARTS} )); then
    echo "[WARN] Training failed with exit=\${TRAIN_EXIT}; requeueing (\${SLURM_RESTART_COUNT:-0}/${MAX_RESTARTS})."
    scontrol requeue "\${SLURM_JOB_ID}" || true
    exit 0
fi

echo "[ERROR] Training failed with exit=\${TRAIN_EXIT}; max restarts reached (\${SLURM_RESTART_COUNT:-0}/${MAX_RESTARTS})."
exit "\${TRAIN_EXIT}"
EOF
    echo "============================================================"
    echo "=== Configuration Summary ==="
    echo "============================================================"
    echo "CLUSTER:          ${CLUSTER}"
    echo "PARTITION:        ${PARTITION}"
    echo "MODEL_VARIANT:    ${MODEL_VARIANT}"
    echo "DATASET_PROVIDER: ${DATASET_PROVIDER}"
    echo "DATA_PATH:        ${DATA_PATH}"
    echo "NODES:            ${NNODES}"
    echo "GPUS_PER_NODE:    ${GPUS_PER_NODE}"
    echo "WORLD_SIZE:       $((NNODES * N_TASKS_PER_NODE))"
    echo "RUN_TIME:         ${RUN_TIME}"
    echo "MAX_RESTARTS:     ${MAX_RESTARTS}"
    echo "REQUEUE_SIGNAL_S: ${REQUEUE_SIGNAL_SECONDS}"
    echo "PRECISION(PR):    ${PR}"
    echo "CHECKPOINT:       ${CHECKPOINT_STORE_PATH}"
    echo "WANDB_RUN_ID_FILE:${CHECKPOINT_STORE_PATH}/wandb_run_id.txt"
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
#SBATCH --job-name=${JOB_NAME}-${ACCOUNT}-${TIMESTAMP}
#SBATCH --output=${SLURM_LOGS}/slurm-%j-${PARTITION}_qwen35-muon.log
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@${REQUEUE_SIGNAL_SECONDS}

set -euo pipefail

export MASTER_ADDR=\$(scontrol show hostnames "\${SLURM_JOB_NODELIST}" | head -n 1)
export WORLD_SIZE=\${SLURM_NTASKS}
echo "MASTER_ADDR=\${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}, WORLD_SIZE=\${WORLD_SIZE}"
echo "SLURM_RESTART_COUNT=\${SLURM_RESTART_COUNT:-0}, MAX_RESTARTS=${MAX_RESTARTS}"

if (( \${SLURM_RESTART_COUNT:-0} >= ${MAX_RESTARTS} )); then
    echo "[ERROR] Restart limit reached before launch: \${SLURM_RESTART_COUNT:-0}/${MAX_RESTARTS}"
    exit 1
fi

TIMEOUT_TRIGGERED=0
SRUN_PID=""
trap 'TIMEOUT_TRIGGERED=1; echo "[WARN] Received USR1 (time limit approaching), will requeue if allowed."; if [[ -n "\${SRUN_PID}" ]] && kill -0 "\${SRUN_PID}" 2>/dev/null; then kill -TERM "\${SRUN_PID}" || true; fi' USR1

set +e
srun \\
    --mpi=pmix -l \\
    --kill-on-bad-exit=1 \\
    --no-container-mount-home \\
    --container-image=${CONTAINER_IMAGE} \\
    --container-mounts=${CONTAINER_MOUNTS} \\
    --container-workdir=${WORKDIR} \\
    bash -lc "set -euo pipefail; \\
        ${TRAINING_CMD}" \\
        2>&1 | tee ${SLURM_LOGS}/\\\${SLURM_JOB_ID}.log &
SRUN_PID=\$!
wait "\${SRUN_PID}"
TRAIN_EXIT=\$?
set -e

if (( TIMEOUT_TRIGGERED == 1 )); then
    if (( \${SLURM_RESTART_COUNT:-0} < ${MAX_RESTARTS} )); then
        echo "[INFO] Requeue on timeout signal (\${SLURM_RESTART_COUNT:-0}/${MAX_RESTARTS})."
        scontrol requeue "\${SLURM_JOB_ID}" || true
        exit 0
    fi
    echo "[ERROR] Timeout signal received but max restarts reached."
    exit 1
fi

if (( TRAIN_EXIT == 0 )); then
    echo "[INFO] Training finished successfully."
    exit 0
fi

if (( \${SLURM_RESTART_COUNT:-0} < ${MAX_RESTARTS} )); then
    echo "[WARN] Training failed with exit=\${TRAIN_EXIT}; requeueing (\${SLURM_RESTART_COUNT:-0}/${MAX_RESTARTS})."
    scontrol requeue "\${SLURM_JOB_ID}" || true
    exit 0
fi

echo "[ERROR] Training failed with exit=\${TRAIN_EXIT}; max restarts reached (\${SLURM_RESTART_COUNT:-0}/${MAX_RESTARTS})."
exit "\${TRAIN_EXIT}"
EOF
    echo "Job submitted successfully!"
fi
set -e
