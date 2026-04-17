IMAGE=/lustre/fsw/general_sa/xshang/sqsh/Pytorch-2512-GB300-HybridEP-V2.sqsh
# IMAGE=/lustre/fsw/general_sa/xshang/sqsh/Pytorch-2512-GB200-HybridEP-Qwen3.5-TE2D.sqsh
TIME=`date "+%Y%m%d"`
# SAVE_NAME=/lustre/fsw/general_sa/xshang/sqsh/Pytorch-2512-GB200-HybridEP-Qwen3.5-TE2D.sqsh

    # --container-save=$SAVE_NAME \
srun -A general_sa -p gb200-backfill -N 1 --exclusive --container-writable \
    --container-image=$IMAGE \
    --container-mounts=/home/xshang:/home/xshang,/lustre/fsw/general_sa/:/lustre/fsw/general_sa/ \
    --container-workdir=/home/xshang/Megatron-LM/examples/multimodal_dev/scripts \
    -t 8:00:00 --pty bash 
