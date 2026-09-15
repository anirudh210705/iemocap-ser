#!/bin/bash
#SBATCH --job-name=ser-bloomz
#SBATCH --partition=gpupart_v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv/bin/activate
python -m ser.train --config configs/bloomz_1b7_main.json
