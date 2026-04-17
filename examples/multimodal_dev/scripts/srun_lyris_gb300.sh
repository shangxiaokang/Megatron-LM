IMAGE=/lustre/fsw/general_sa/xshang/sqsh/Pytorch-2512-GB300-HybridEP-V2.sqsh
TIME=`date "+%Y%m%d"`
SAVE_NAME=/lustre/fsw/general_sa/xshang/sqsh/Pytorch-2512-GB300-HybridEP-Qwen3.5-TE2D.sqsh

srun -A general_sa -p gb300-backfill -N 1 --exclusive --container-writable \
    --container-image=$IMAGE \
    --container-save=$SAVE_NAME \
    --container-mounts=/home/xshang:/home/xshang,/lustre/fsw/general_sa/:/lustre/fsw/general_sa/ \
    --container-workdir=/home/xshang/my-script \
    -t 5:00:00 --pty bash 
