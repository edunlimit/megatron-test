# Copyright (c) 2025 Alibaba PAI and Nvidia Megatron-LM Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import subprocess
import sys
from pathlib import Path
import datetime
import torch
import torch._dynamo

# ########################################
# # SageMaker Revised
# os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
# os.environ['LOCAL_RANK'] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
# local_rank = int(os.environ['LOCAL_RANK'])
# os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
# world_rank = int(os.environ['RANK'])
# os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
# world_size = int(os.environ['WORLD_SIZE'])

# # For g5.48xlarge, there are 8 GPUs per node
# gpus_per_node = 8
# os.environ['NODE_RANK'] = str(world_rank // gpus_per_node)
# node_rank = int(os.environ['NODE_RANK'])

# # Disable P2P for g5.48xlarge since it doesn't support it
# os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IB_DISABLE"] = "1"
# os.environ["NCCL_DEBUG_SUBSYS"]="ALL"

# os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(gpus_per_node))

# # Setting up AWS credentials directory
# os.makedirs('/root/.aws/', exist_ok=True)

# set_path = '/opt/ml/code/config'
# file = Path(set_path)
# if file.exists() and os.environ['LOCAL_RANK'] == '0':
#     subprocess.run(['cp', '-r', set_path, '/root/.aws/'])
# ########################################

# # Initialize the process group
# torch.distributed.init_process_group(
#     backend="nccl",
#     init_method="env://",
#     timeout=datetime.timedelta(minutes=30)  # Increase timeout
# )
# device = torch.device(f"cuda:{local_rank}")
# store = torch.distributed.TCPStore("localhost", 29500, 2, True)
# print("TCP store initialized successfully")

# # Verify process group with simple all_reduce
# try:
#     tensor = torch.ones(1, device=device)
#     torch.distributed.all_reduce(tensor)
#     torch.cuda.synchronize()  # Ensure the operation is complete
#     print(f"All-reduce test on rank {world_rank}: {tensor.item()} (should be {world_size})")
# except Exception as e:
#     print(f"Error during all-reduce on rank {world_rank}: {e}")

# torch.distributed.barrier()
# print(f"Passed barrier on rank {world_rank}")
# ########################################
# ## SageMaker Revised

from typing import Union

from megatron.core.enums import ModelType
from megatron.training import get_args, pretrain, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args

from megatron_patch.arguments import get_patch_args
from megatron_patch.model.qwen2.layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron_patch.model.qwen2.model import GPTModel
from megatron_patch.model.qwen2.transformer_config import Qwen2TransformerConfig
from megatron_patch.tokenizer import build_tokenizer, get_tokenizer
from megatron_patch.data import train_valid_test_datasets_provider

torch._dynamo.config.suppress_errors = True


def model_provider(
    pre_process=True, post_process=True
) -> Union[GPTModel]:

    args = get_args()
    build_tokenizer(args)
    print_rank_0("building qwen2 model ...")

    config = core_transformer_config_from_args(args, Qwen2TransformerConfig)
    use_te = args.transformer_impl == "transformer_engine"

    if use_te:
        print_rank_0("building qwen2 model in TE...")
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            args.num_experts, args.moe_grouped_gemm, args.qk_layernorm
        )
    else:
        print_rank_0("building qwen2 model in Mcore...")
        transformer_layer_spec = get_gpt_layer_local_spec(
            args.num_experts, args.moe_grouped_gemm, args.qk_layernorm
        )

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
    )

    return model


if __name__ == "__main__":
    from megatron_patch.template.helper import forward_step
    train_valid_test_datasets_provider.is_distributed = True
    
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=get_patch_args,
    )
