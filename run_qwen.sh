#!/bin/bash
#SBATCH --job-name=dpo_training
#SBATCH --partition=a100                   # GPU partition
#SBATCH --qos=a100_aixpert                 # QoS for AIXpert users
#SBATCH --gres=gpu:a100:1                  # Request 1 A100 GPU
#SBATCH --time=06:00:00                    # Max walltime
#SBATCH --cpus-per-task=16                 # Number of CPU cores
#SBATCH --mem=32G                          # Total memory
#SBATCH --output=slurm-%j.out              # STDOUT (%j = JobID)
#SBATCH --error=slurm-%j.err               # STDERR (%j = JobID)

set -euo pipefail

# Hugging Face cache (100GB area)
export HF_HOME=/scratch/ssd004/scratch/sindchad/model-weights
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/dataset

# Hugging Face authentication (safe method)
export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN:-""}
if [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo "⚠️  Warning: HF_TOKEN not found in environment. If using private models, run:"
    echo "    export HF_TOKEN=your_hf_token"
    echo "before submitting the job."
fi
# Ensure logs dir exists (optional, since output goes to slurm-%j.* files now)
mkdir -p logs

# Go to your project
cd /projects/aixpert/users/sindhu/Con-J/src

# Run with uv
uv run python -u dpo_training.py
