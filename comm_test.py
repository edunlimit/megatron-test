#!/usr/bin/env python3
import os
import sys
import time
import socket
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

def setup_distributed(backend='nccl'):
    """
    Initialize the distributed environment using Slurm environment variables
    """
    # Slurm variables
    rank = int(os.environ.get('SLURM_PROCID', '0'))
    world_size = int(os.environ.get('SLURM_NTASKS', '1'))
    local_rank = int(os.environ.get('SLURM_LOCALID', '0'))

    # Get the master node info from SLURM
    master_addr = os.environ.get('SLURM_LAUNCH_NODE_IPADDR')
    if not master_addr:
        # If not available, use the first node in the nodelist
        nodes = os.environ.get('SLURM_NODELIST', '')
        if nodes:
            master_addr = nodes.split(',')[0]
        else:
            # Fallback to localhost if not running in Slurm
            master_addr = 'localhost'

    # Use a fixed port for communication
    master_port = '29500'

    # Set environment variables for PyTorch distributed
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    # Print debug info
    print(f"Initializing process group: rank={rank}, world_size={world_size}, "
          f"local_rank={local_rank}, master={master_addr}:{master_port}")

    # Initialize the process group
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    # Set the device for this process
    torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank

def cleanup():
    """
    Clean up the distributed environment
    """
    if dist.is_initialized():
        dist.destroy_process_group()

def run_test():
    """
    Run a simple distributed test
    """
    try:
        # Setup distributed environment
        rank, world_size, local_rank = setup_distributed()

        # Get node information
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        gpu_name = torch.cuda.get_device_name(local_rank)
        gpu_count = torch.cuda.device_count()

        # Print node information
        print(f"Rank {rank}/{world_size} | Local rank: {local_rank} | "
              f"Host: {hostname} ({ip_address}) | "
              f"GPU: {gpu_name} | GPU Count: {gpu_count}")

        # Create a simple tensor on the GPU
        tensor = torch.ones(1).to(local_rank) * rank

        # All-reduce operation to test communication
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        expected_sum = sum(range(world_size))

        # Verify the result
        if tensor.item() == expected_sum:
            print(f"Rank {rank}: All-reduce successful! "
                  f"Result: {tensor.item()}, Expected: {expected_sum}")
        else:
            print(f"Rank {rank}: All-reduce FAILED! "
                  f"Result: {tensor.item()}, Expected: {expected_sum}")

        # Test GPU performance
        start_time = time.time()
        size = 4096  # Matrix size
        a = torch.randn(size, size, device=f'cuda:{local_rank}')
        b = torch.randn(size, size, device=f'cuda:{local_rank}')

        # Synchronize before timing
        torch.cuda.synchronize()
        mm_start = time.time()

        # Matrix multiplication
        c = torch.matmul(a, b)

        # Synchronize after operation
        torch.cuda.synchronize()
        mm_end = time.time()

        print(f"Rank {rank}: Matrix multiplication ({size}x{size}) took "
              f"{mm_end - mm_start:.4f} seconds")

        # Test network bandwidth between GPUs
        if world_size > 1:
            # Only test if we have multiple processes
            tensor_sizes = [1024, 10240, 102400, 1024000, 10240000]

            for size in tensor_sizes:
                # Create data
                data = torch.ones(size, device=f'cuda:{local_rank}') * rank

                # Synchronize before timing
                torch.cuda.synchronize()
                dist.barrier()
                start = time.time()

                # All-reduce
                dist.all_reduce(data, op=dist.ReduceOp.SUM)

                # Synchronize after operation
                torch.cuda.synchronize()
                end = time.time()

                # Calculate bandwidth (bytes/s)
                bytes_transferred = 4 * size * 2  # 4 bytes per float, 2 for bidirectional
                bandwidth = bytes_transferred / (end - start) / 1e9  # GB/s

                print(f"Rank {rank}: All-reduce bandwidth for {size} elements: "
                      f"{bandwidth:.2f} GB/s ({end - start:.6f} seconds)")

        # Wait for all processes to complete
        dist.barrier()
        if rank == 0:
            print("All tests completed successfully!")

    except Exception as e:
        print(f"Error in rank {os.environ.get('SLURM_PROCID', '0')}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        cleanup()

if __name__ == "__main__":
    run_test()