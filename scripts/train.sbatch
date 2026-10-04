#!/usr/bin/env bash
#SBATCH --job-name=p2n-native
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
set -euo pipefail
export GPUS=8 OMP_NUM_THREADS=4
exec bash "${SLURM_SUBMIT_DIR:-$PWD}/scripts/train.sh" "${METHOD:?Set METHOD to vanilla or p2n}" "$@"
