#!/bin/bash
#SBATCH --job-name=ser-w2v-mlp
#SBATCH --partition=gpupart_v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv/bin/activate
python -m ser.train --config configs/wav2vec2_base.json
