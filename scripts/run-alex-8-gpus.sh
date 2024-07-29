#!/bin/bash
#SBATCH --job-name=wiki-en               # Job name
#SBATCH --output=logs/%j.out             # Output file
#SBATCH --error=logs/%j.err              # Error file
#SBATCH --gres=gpu:a100:8 -C a100_80
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --time=8:00:00                   # Walltime limit (hh:mm:ss)


module add python
module add gcc/12.1.0
module add cuda/12.1.1

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

HF_HOME="$TMPDIR/.cache"
WANDB_CACHE_DIR="$TMPDIR/.wandb_cache"
WANDB_DATA_DIR="$TMPDIR/.wandb_staging"

mkdir -p "$HF_HOME"
mkdir -p "$WANDB_CACHE_DIR"
mkdir -p "$WANDB_DATA_DIR"

export HF_HOME
export WANDB_CACHE_DIR
export WANDB_DATA_DIR

export TOKENIZERS_PARALLELISM=false

venv_path=".env"
source "$venv_path/bin/activate"

export WANDB_PROJECT="wiki-en"
export WANDB_GROUP="7b"
export WANDB_JOB_TYPE="pretraining"

mkdir $FASTTMP/llm-output

srun accelerate launch --mixed_precision bf16 train.py --output-dir $FASTTMP/llm-output --cache-dir $TMPDIR/.cache --proxy

deactivate
