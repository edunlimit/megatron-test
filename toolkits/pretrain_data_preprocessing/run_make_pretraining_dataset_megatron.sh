#! /bin/bash
#SBATCH --ntasks=16                 # Total number of tasks (8 GPUs × 2 nodes)
#SBATCH --nodes=2                   # Use 2 nodes
#SBATCH --ntasks-per-node=8         # 8 tasks per node (one per GPU)
#SBATCH --partition=train           # Partition to submit the job to
#SBATCH --cpus-per-task=8           # Number of CPU cores per task
#SBATCH --gres=gpu:8                # 8 GPUs per node
#SBATCH --output=data_%j.log

START_TIME=$SECONDS

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export PYTHONPATH=$PYTHONPATH:${MEGATRON_PATH}:${MEGATRON_PATH}/Megatron-LM-240405

input_data_dir=$1
tokenizer=$2
json_keys=$3
output_data_dir=$4
load_dir=$5

INPUT="${input_data_dir}"
source /opt/pytorch/bin/activate

if [ $tokenizer = "Qwen2Tokenizer" ]; then
  uv run python preprocess_data_megatron.py \
  --input ${INPUT} \
  --output-prefix ${output_data_dir}/mmap_qwen2_datasets \
  --patch-tokenizer-type Qwen2Tokenizer \
  --json-keys ${json_keys} \
  --load ${load_dir} \
  --workers 2 \
  --partitions 2 \
  --keep-sequential-samples \
  --append-eod

elif [ $tokenizer = "DeepSeekV2Tokenizer" ]; then
  python preprocess_data_megatron.py \
  --input ${INPUT} \
  --output-prefix ${output_data_dir}/mmap_deepseekv2_datasets \
  --patch-tokenizer-type DeepSeekV2Tokenizer \
  --json-keys ${json_keys} \
  --load ${load_dir} \
  --workers 8 \
  --partitions 1 \
  --keep-sequential-samples \
  --append-eod

elif [ $tokenizer = "LLamaTokenizer" ]; then
  python preprocess_data_megatron.py \
  --input ${INPUT} \
  --output-prefix ${output_data_dir}/mmap_llama_datasets \
  --patch-tokenizer-type LLamaTokenizer \
  --json-keys ${json_keys} \
  --load ${load_dir} \
  --workers 16 \
  --partitions 1 \
  --keep-sequential-samples \
  --append-eod
  
elif [ $tokenizer = "LLama2Tokenizer" ]; then
  python preprocess_data_megatron.py \
  --input ${INPUT} \
  --output-prefix ${output_data_dir}/mmap_llama2_datasets \
  --patch-tokenizer-type LLama2Tokenizer \
  --json-keys ${json_keys} \
  --load ${load_dir} \
  --workers 16 \
  --partitions 1 \
  --keep-sequential-samples \
  --append-eod

elif [ $tokenizer = "LLama3Tokenizer" ]; then
  python preprocess_data_megatron.py \
  --input ${INPUT} \
  --output-prefix ${output_data_dir}/mmap_llama3_datasets \
  --patch-tokenizer-type LLama3Tokenizer \
  --load ${load_dir} \
  --workers 16 \
  --partitions 1 \
  --keep-sequential-samples \
  --append-eod

fi

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
