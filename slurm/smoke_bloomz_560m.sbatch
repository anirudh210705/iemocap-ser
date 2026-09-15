#!/bin/bash
#SBATCH --job-name=ser-smoke
#SBATCH --partition=gpupart_1hour
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv/bin/activate
python -m ser.smoke_test --config configs/bloomz_560m_smoke.json --max-seconds 4
