#!/bin/bash
#SBATCH --job-name=train_bert            # Job name
#SBATCH --output=logs/%j.out             # Output file
#SBATCH --error=logs/%j.err              # Error file
#SBATCH --clusters=wice                  # Cluster
#SBATCH --partition=gpu                  # Specify the partition name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks (processes) per node
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=2                # Number of tasks (processes) per node
#SBATCH --time=6:00:00                   # Walltime limit (hh:mm:ss)


HF_HOME="/cw/dtaijupiter/NoCsBack/dtai/pieterd/hf_cache"
WANDB_CACHE_DIR="/cw/dtaijupiter/NoCsBack/dtai/pieterd/.wandb_cache"
WANDB_DATA_DIR="/cw/dtaijupiter/NoCsBack/dtai/pieterd/.wandb_staging"

export HF_HOME
export WANDB_CACHE_DIR
export WANDB_DATA_DIR

export TOKENIZERS_PARALLELISM=false

venv_path=".env"

source "$venv_path/bin/activate"

export WANDB_PROJECT="wiki-en"
export WANDB_GROUP="7b"
export WANDB_JOB_TYPE="pretraining"

mkdir llm-output

python train.py --output-dir llm-output/

deactivate
