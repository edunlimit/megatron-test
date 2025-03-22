#!/bin/bash
#SBATCH --ntasks=16                 # Total number of tasks (8 GPUs × 2 nodes)
#SBATCH --nodes=2                   # Use 2 nodes
#SBATCH --ntasks-per-node=8         # 8 tasks per node (one per GPU)
#SBATCH --partition=train           # Partition to submit the job to
#SBATCH --cpus-per-task=8           # Number of CPU cores per task
#SBATCH --gres=gpu:8                # 8 GPUs per node
#SBATCH --job-name=distributed_train  # Job name
#SBATCH --output=train_%j.log

set -e
ENV=$1
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATH}/PAI-Megatron-LM-240718:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Debug information
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "MEGATRON_PATH= "MEGATRON_PATH
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

# Load environment
source ~/.bashrc

export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Set DDP variables
export MASTER_PORT=29500
export WORLD_SIZE=$(( 8 * $SLURM_JOB_NUM_NODES ))
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)

echo "MASTER_PORT=$MASTER_PORT"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "MASTER_ADDR=$MASTER_ADDR"

# Dataset config
if [ -z ${MP_DATASET_TYPE} ]; then
    MP_DATASET_TYPE="idxmap"
fi

# Megatron options (same as before)
megatron_options="  \
        --save ${SAVED_PRETRAIN_CHECKPOINT_PATH} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --clip-grad 1.0 \
        --init-method-std 0.008 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --lr-decay-iters ${LR_DECAY_ITERS} \
        --lr-warmup-iters ${LR_WARMUP_ITERS} \
        --train-iters ${TRAIN_ITERS} \
        --micro-batch-size ${BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --ffn-hidden-size ${INTERMEDIATE_SIZE} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
        --max-padding-length ${PAD_LEN} \
        --log-interval 1 \
        --log-throughput \
        --eval-interval 10000 \
        --eval-iters 10 \
        --save-interval ${SAVE_INTERVAL} \
        --tensorboard-queue-size 1 \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --context-parallel-size ${CP} \
        --no-load-optim \
        --no-load-rng \
        --num-workers 8 \
        --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
        --patch-tokenizer-type Qwen2Tokenizer \
        --swiglu \
        --normalization RMSNorm \
        --norm-epsilon ${RMS_NORM_EPS} \
        --use-rotary-position-embeddings \
        --position-embedding-type rope \
        --disable-bias-linear \
        --add-qkv-bias \
        --rotary-percent 1.0 \
        --rotary-base 1000000 \
        --rotary-seq-len-interpolation-factor 1 \
        --no-save-optim \
        "

# Run with srun directly
run_cmd="srun python /fsx/dataset/megatron-test/examples/qwen2/pretrain_qwen.py \
 ${megatron_options} ${dataset_option} ${pr_options} ${load_options} ${te_options} ${activation_checkpoint_options} \
 ${do_options} ${sp_options} ${gqa_options} ${offload_option} ${comm_overlap_option} ${sft_option}  ${tie_option} ${vp_options} ${packing_options}"

echo ${run_cmd}
eval ${run_cmd}
set +x