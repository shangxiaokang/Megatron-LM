pip install flask flask-restful uvloop
CODE_DIR=/home/xshang/Megatron-LM
OUT_DIR=/home/xshang/Megatron-LM/examples/multimodal_dev/scripts/output
TOKENIZER_MODEL=moonshotai/Moonlight-16B-A3B-Instruct
CHECKPOINT=/lustre/fsw/general_sa/xshang/huggingface/Moonlight-16B
DATA_PATH=/lustre/fsw/general_sa/xshang/dataset/imdb/imdb_megatron_text_document
DATA_ARGS="
    --data-path $DATA_PATH
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model $TOKENIZER_MODEL
    --trust-remote-code
    --tiktoken-pattern v2
    --split 90,5,5
    --num-workers 6
    --no-create-attention-mask-in-dataloader
"

# Set the parallel startegy env and config
GBS=768
MBS=1
EP_SIZE=4
TP_SIZE=1
PP_SIZE=1
CP_SIZE=1

# original model configs https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json
# VPP: to use --num-layers-per-virtual-pipeline-stage ${VPP_SIZE} for large MoE models

# === COMMUNICATION OVERLAP OPTIMIZATION ===
# 1. EP (Expert Parallel) Communication Overlap via PP:
#    --delay-wgrad-compute --overlap-moe-expert-parallel-comm
#    Requires: PP_size and EP_SIZE > 1
# 2. PP (Pipeline Parallel) Communication Overlap:
#    Requires interleaved schedule to enable: --num-layers-per-virtual-pipeline-stage
#    --overlap-p2p-communication-warmup-flush enables overlap in warmup/flush phases
#    Overlaps p2p communication with computation during pipeline execution
#    When overlap enabled, batch_p2p_comm is automatically disabled
# 3. Compatibility Notes:
#    - EP overlap cannot be used with full recomputation or shared expert overlap
#    - PP overlap incompatible with batch_p2p_comm (automatically handled)
#    - TP overlap needs CUDA_DEVICE_MAX_CONNECTIONS=1, conflicts with EP overlap
#    - Priority: EP overlap > PP overlap > TP overlap (for MoE models)
# expert tensor parallel: --expert-tensor-parallel-size


    # --overlap-param-gather
MODEL_PARALLEL_ARGS="
    --distributed-timeout-minutes 120
    --micro-batch-size ${MBS}
    --global-batch-size ${GBS}
    --expert-model-parallel-size ${EP_SIZE}
    --tensor-model-parallel-size ${TP_SIZE}
    --pipeline-model-parallel-size ${PP_SIZE}
    --context-parallel-size ${CP_SIZE}
    --overlap-grad-reduce

    --num-experts 64
    --moe-layer-freq 1
    --moe-ffn-hidden-size 1408
    --moe-shared-expert-intermediate-size 2816
    --moe-router-load-balancing-type seq_aux_loss
    --moe-aux-loss-coeff 1e-3
    --moe-router-topk 6
    --moe-router-pre-softmax
    --moe-grouped-gemm
    --moe-router-dtype fp32

    --moe-router-topk-scaling-factor 2.446
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 1e-3
    --moe-token-dispatcher-type alltoall

"
#    --sequence-parallel
#    --overlap-param-gather
#    --overlap-grad-reduce

# Set the model env and config
# To use --te-rng-tracker --external-cuda-graph --cuda-graph-scope attn, ...
# CUDA Graph for MoE: --cuda-graph-scope includes moe_router,moe_preprocess to capture MoE layers
# To use --moe-router-fusion, need to use latest TE
# DeepEP: --moe-token-dispatcher-type flex --moe-enable-deepep --moe-deepep-num-sms 20
# --cuda-graph-warmup-steps 3: number of warmup steps before capturing graphs
# --cuda-graph-use-single-mempool: use single memory pool for better memory efficiency
# cuda graph cannot capture moe_router in force balance mode, and moe_preprocess cuda graph is only supported with moe_router cuda graph


SEQ_LENGTH=8192
    # --use-checkpoint-args
    # --no-use-tokenizer-model-from-checkpoint-args
MODEL_ARGS="
    --untie-embeddings-and-output-weights
    --no-bias-swiglu-fusion
    --swiglu
    --use-mcore-models
    --transformer-impl transformer_engine
    --position-embedding-type rope
    --no-rope-fusion
    --rotary-percent 1.0
    --rotary-base 50000
    --normalization RMSNorm
    --norm-epsilon 1e-5
    --multi-latent-attention
    --num-attention-heads 16
    --attention-backend auto
    --hidden-dropout 0.0
    --attention-dropout 0.0
    --ckpt-format torch_dist
    --disable-bias-linear
    --untie-embeddings-and-output-weights
    --kv-lora-rank 512
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
    --seq-length ${SEQ_LENGTH}
    --num-layers 27
    --hidden-size 2048
    --qk-layernorm
    --max-position-embeddings ${SEQ_LENGTH}
    --cuda-graph-impl transformer_engine
    --cuda-graph-scope attn moe_router
"

PR=${PR:-bf16}

WANDB_PROJECT='Moonlight-16B'
EXP_NAME="Moonlight-16B_tp${TP_SIZE}_ep${EP_SIZE}_pp${PP_SIZE}_${PR}_MBS${MBS}_GBS${GBS}_muon"
# Set the training time and log info
EVAL_AND_LOGGING_ARGS="
    --train-samples 133632768
    --log-interval 1
    --save-interval 100
    --eval-interval 1000
    --eval-iters 1
    --log-throughput
    --timing-log-level 0
    --tensorboard-log-interval 1
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --log-params-norm
    --log-num-zeros-in-grad
    --log-validation-ppl-to-tensorboard
    --tensorboard-dir $OUT_DIR
    --wandb-project "$WANDB_PROJECT"
    --wandb-exp-name "$EXP_NAME"
    --wandb-save-dir "$CHECKPOINT"
"
    # --use-precision-aware-optimizer
# Set the optimizer
TRAINING_ARGS="
    --dist-ckpt-strictness raise_unexpected
    --no-load-optim
    --no-load-rng
    --use-distributed-optimizer
    --main-grads-dtype fp32
    --main-params-dtype fp32
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl native
    --lr 0.00015
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --optimizer muon --muon-momentum 0.95 --muon-scale-mode spectral --muon-extra-scale-factor 0.2 --muon-no-split-qkv
    --clip-grad 1.0
    --lr-decay-style WSD
    --lr-wsd-decay-style linear
    --lr-wsd-decay-samples 5377
    --lr-warmup-samples 4000
    --min-lr 1.0e-5
    --manual-gc
    --manual-gc-interval 10
    --no-create-attention-mask-in-dataloader
"
    # --exp-avg-dtype bf16
    # --exp-avg-sq-dtype bf16
    #--load $CHECKPOINT
# Set the training precision
# For FP8: --fp8-format e4m3, --fp8-recipe mxfp8, --fp8_param_gather, --reuse-grad-buf-for-mxfp8-param-ag
PRECISION_ARGS="
    --bf16
"

PR_ARGS=()
if [[ ${PR} == "mxfp8" ]]; then
  PR_ARGS="
    --fp8-recipe mxfp8
    --fp8-format e4m3
    --moe-router-padding-for-quantization
    --overlap-grad-reduce
  "
fi
    # --overlap-param-gather
    # --fp8-param-gather
    # --reuse-grad-buf-for-mxfp8-param-ag
    # --use-precision-aware-optimizer
    # --main-grads-dtype fp32
    # --main-params-dtype fp32
    # --exp-avg-dtype bf16
    # --exp-avg-sq-dtype bf16

# Training command with proper redirection
# exec 2&1>$OUT_DIR/${NAME}_TP${TP_SIZE}PP${PP_SIZE}EP${EP_SIZE}MBS${MBS}.log 2>$OUT_DIR/${NAME}_TP${TP_SIZE}PP${PP_SIZE}EP${EP_SIZE}MBS${MBS}.err

# ${CODE_DIR}/bindpcie --cpu=node python3 ${CODE_DIR}/pretrain_gpt.py \
cmd="torchrun --nproc_per_node 4 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6000 \
    ${CODE_DIR}/pretrain_gpt.py \
    $DATA_ARGS \
    $MODEL_PARALLEL_ARGS \
    $MODEL_ARGS \
    $PR_ARGS \
    $EVAL_AND_LOGGING_ARGS \
    $TRAINING_ARGS \
    $PRECISION_ARGS \
    --distributed-backend nccl \
    --auto-detect-ckpt-format"
eval $cmd
