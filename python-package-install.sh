#!/bin/bash

#SBATCH --ntasks=2                  # Number of tasks (one per node)
#SBATCH --nodes=2                   # Use 2 nodes
#SBATCH --time=01:00:00             # Timeout for the job
#SBATCH --mem=4GB                   # Memory per node
#SBATCH --partition=train         # Partition to submit the job to


source /opt/parallelcluster/pyenv/versions/3.9.20/envs/awsbatch_virtualenv/bin/activate

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
packaging \
ninja \

pip3 install --verbose --no-cache-dir transformer_engine[pytorch]


sudo chown -R ec2-user:ec2-user /home/ec2-user/apex

sudo mkdir -p apex \
&& sudo chmod -R 777 apex \
&& cd apex \
&& git clone https://github.com/NVIDIA/apex.git . \
&& pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./ \
&& cd .. \
&& rm -rf /apex

echo "Python modules and dependencies have been successfully installed."