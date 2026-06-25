#!/bin/bash
#SBATCH --job-name=hmm_cv
#SBATCH --output=logs/hmm_cv_%j.out
#SBATCH --error=logs/hmm_cv_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=0

set -euo pipefail

cd /mnt/home/srafilson/code/sequences

module load python
source .venv/bin/activate

mkdir -p logs results_20260618/full_fit

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

disBatch tasks/hmm_tasks.txt