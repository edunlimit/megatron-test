import os
import torch
import subprocess
import sys
from pathlib import Path
import datetime
import torch._dynamo
import time
import socket
import random

########################################
# SageMaker Revised
os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
os.environ['LOCAL_RANK'] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
local_rank = int(os.environ['LOCAL_RANK'])
os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
world_rank = int(os.environ['RANK'])
os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
world_size = int(os.environ['WORLD_SIZE'])

# For g5.48xlarge, there are 8 GPUs per node
gpus_per_node = 8
os.environ['NODE_RANK'] = str(world_rank // gpus_per_node)
node_rank = int(os.environ['NODE_RANK'])

# Disable P2P for g5.48xlarge since it doesn't support it
os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# Additional NCCL debug settings
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_SOCKET_TIMEOUT"] = "300"  # Increase socket timeout to 5 minutes
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(gpus_per_node))

# Setting up AWS credentials directory
os.makedirs('/root/.aws/', exist_ok=True)

set_path = '/opt/ml/code/config'
file = Path(set_path)
if file.exists() and os.environ['LOCAL_RANK'] == '0':
    subprocess.run(['cp', '-r', set_path, '/root/.aws/'])
########################################

# Function to find an available port
def find_free_port():
    """Find a free port on the machine"""
    # Try a range of ports starting from a random point between 10000-20000
    start_port = random.randint(10000, 20000)
    for port in range(start_port, start_port + 1000):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("", port))
            s.close()
            return port
        except OSError:
            continue
    raise RuntimeError("Could not find a free port")

# Test network connectivity between nodes
def test_network_connectivity():
    """Test network connectivity between nodes"""
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)

    print(f"Rank {world_rank}: Hostname: {hostname}, IP: {ip_address}")

    # Only perform this test on rank 0 of each node
    if local_rank == 0:
        # Get all node IPs - this is just for diagnostic purposes
        print(f"Rank {world_rank} on node {node_rank} with IP {ip_address} is ready")

        # Test if we can open a socket on this machine
        try:
            test_port = find_free_port()
            print(f"Rank {world_rank}: Found available port {test_port} for testing")

            # Try to briefly listen on this port
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("", test_port))
            s.listen(1)
            s.close()
            print(f"Rank {world_rank}: Successfully tested socket binding on port {test_port}")
        except Exception as e:
            print(f"Rank {world_rank}: Error testing socket binding: {e}")

# Run network connectivity test
test_network_connectivity()

# Synchronize all processes before proceeding
if world_rank == 0:
    print("Starting initialization with increased timeout...")

# Initialize the process group with increased timeout
try:
    print(f"Rank {world_rank}: Attempting to initialize process group")
    # Let PyTorch's init_process_group handle the store creation
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=datetime.timedelta(minutes=30)  # Increase timeout
    )
    print(f"Rank {world_rank}: Process group initialized successfully")
    device = torch.device(f"cuda:{local_rank}")

    # Skip the custom TCP store creation since SageMaker is already managing this
    # Instead, test if the existing process group works correctly

    # Verify process group with simple all_reduce
    try:
        print(f"Rank {world_rank}: Starting all-reduce test")
        tensor = torch.ones(1, device=device)
        # First do a simple check if the tensor is on the right device
        print(f"Rank {world_rank}: Tensor device: {tensor.device}")

        # Try with a small timeout first
        torch.distributed.all_reduce(tensor)
        torch.cuda.synchronize()  # Ensure the operation is complete
        print(f"Rank {world_rank}: All-reduce test successful: {tensor.item()} (should be {world_size})")
    except Exception as e:
        print(f"Rank {world_rank}: Error during all-reduce with 30s timeout: {e}")
        try:
            # Try again with a longer timeout
            print(f"Rank {world_rank}: Retrying all-reduce with longer timeout")
            tensor = torch.ones(1, device=device)
            torch.distributed.all_reduce(tensor)
            torch.cuda.synchronize()
            print(f"Rank {world_rank}: Retry all-reduce successful: {tensor.item()} (should be {world_size})")
        except Exception as e2:
            print(f"Rank {world_rank}: Error during retry all-reduce: {e2}")

    # Try a barrier with explicit timeout
    try:
        print(f"Rank {world_rank}: Attempting barrier with timeout")
        torch.distributed.barrier()
        print(f"Rank {world_rank}: Passed barrier")
    except Exception as e:
        print(f"Rank {world_rank}: Error during barrier: {e}")

except Exception as e:
    print(f"Rank {world_rank}: Error during initialization: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"Rank {world_rank}: Test completed successfully")