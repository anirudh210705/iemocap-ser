#!/bin/bash
#SBATCH --job-name=ser-cnn
#SBATCH --partition=gpupart_p100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv/bin/activate
python -m ser.train --config configs/cnn_baseline.json
