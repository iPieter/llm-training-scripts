#!/bin/bash
#SBATCH --job-name=train_bert            # Job name
#SBATCH --output=logs/%j.out             # Output file
#SBATCH --error=logs/%j.err              # Error file
#SBATCH --clusters=wice                  # Cluster
#SBATCH --partition=gpu                  # Specify the partition name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks (processes) per node
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=4                # Number of tasks (processes) per node
#SBATCH --time=4:00:00                   # Walltime limit (hh:mm:ss)

# Set HF_HOME if VSC_SCRATCH_SITE doesn't exist
if [ -n "$VSC_SCRATCH_SITE" ]; then
  HF_HOME="$VSC_SCRATCH_SITE/.cache"
  WANDB_CACHE_DIR="$VSC_SCRATCH_SITE/.wandb_cache"
  WANDB_DATA_DIR="$VSC_SCRATCH_SITE/.wandb_staging"
else
  HF_HOME="/cw/dtaijupiter/NoCsBack/dtai/pieterd/hf_cache"
  WANDB_CACHE_DIR="/cw/dtaijupiter/NoCsBack/dtai/pieterd/.wandb_cache"
  WANDB_DATA_DIR="/cw/dtaijupiter/NoCsBack/dtai/pieterd/.wandb_staging"
fi

mkdir -p "$HF_HOME"
mkdir -p "$WANDB_CACHE_DIR"
mkdir -p "$WANDB_DATA_DIR"

export HF_HOME
export WANDB_CACHE_DIR
export WANDB_DATA_DIR

export TOKENIZERS_PARALLELISM=false

venv_path=".env"

# Check if .env folder already exists
if [ ! -d "$venv_path" ]; then
  # .env folder does not exist, create and activate a new virtual environment
  conda activate py310-base
  python3 -m venv "$venv_path"
  source "$venv_path/bin/activate"

  # Install Python packages from requirements.txt
  pip install -r requirements.txt
else
  # .env folder already exists, activate the existing virtual environment
  source "$venv_path/bin/activate"
fi


# Load nccl stuff
ml NCCL/2.10.3-GCCcore-10.3.0-CUDA-11.3.1 
ml CUDA/11.7.1

#ulimit -c 0conda env remove --name 

export WANDB_PROJECT="tiktotok-nl"
export WANDB_GROUP="bpe"

srun accelerate launch train.py

deactivate
