#!/bin/bash

#SBATCH --ntasks=2                  # Number of tasks (one per node)
#SBATCH --nodes=2                   # Use 2 nodes
#SBATCH --time=01:00:00             # Timeout for the job
#SBATCH --mem=4GB                   # Memory per node
#SBATCH --partition=train         # Partition to submit the job to

srun -N 2 -n 2 bash -c '
	#source /opt/parallelcluster/pyenv/versions/3.9.20/envs/awsbatch_virtualenv/bin/activate

	sudo chown -R ec2-user:ec2-user /opt/parallelcluster/pyenv/versions/3.9.20/envs/awsbatch_virtualenv/

	pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

	pip install --no-cache-dir  \
	regex \
	einops \
	flask-restful \
	nltk \
	pytest \
	transformers \
	pytest_asyncio \
	pytest-cov \
	pytest_mock \
	pytest-random-order \
	sentencepiece \
	tiktoken \
	wrapt \
	zarr \
	wandb \
	datasets \
	wheel \
	ninja \
	pybind11 \
	megatron-core \
	tensorboard \
    importlib-metadata=6.9

	pip install --upgrade --force-reinstall pip setuptools packaging
	pip install --upgrade importlib-metadata
'

srun -N 2 -n 2 bash -c '
	#source /opt/parallelcluster/pyenv/versions/3.9.20/envs/awsbatch_virtualenv/bin/activate
	sudo yum install python3-devel -y
'

srun --ntasks=2 --nodes=2 bash -c "
    #source /opt/parallelcluster/pyenv/versions/3.9.20/envs/awsbatch_virtualenv/bin/activate
    export MAX_JOBS=4
    env | grep MAX_JOBS
    echo "MAX_JOBS="$MAX_JOBS
    pip -v install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.6/flash_attn-2.6.3+cu126torch2.6-cp312-cp312-linux_x86_64.whl
    export MAX_JOBS=48
    env | grep MAX_JOBS
    echo "Changing back to MAX_JOBS="$MAX_JOBS
"

srun -N 2 -n 2 bash -c '
	#source /opt/parallelcluster/pyenv/versions/3.9.20/envs/awsbatch_virtualenv/bin/activate
	#pip3 install --verbose --no-cache-dir transformer_engine[pytorch]==1.13.0
	pip3 install --verbose --no-cache-dir transformer_engine[pytorch]
'


echo "Python modules and dependencies have been successfully installed."
