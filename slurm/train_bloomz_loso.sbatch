#!/bin/bash
#SBATCH --job-name=ser-bloomz-loso
#SBATCH --partition=gpupart_v100,gpupart_p100
#SBATCH --array=1-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-23:00:00
#SBATCH --output=logs/%x-fold%A_%a.out
#SBATCH --error=logs/%x-fold%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv/bin/activate

# run_folds automatically resumes from fold_N/last.pt when it exists.
python -m ser.run_folds \
  --config configs/bloomz_1b7_main.json \
  --folds "${SLURM_ARRAY_TASK_ID}"
