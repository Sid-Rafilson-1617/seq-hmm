#!/bin/bash
#SBATCH --job-name=hmm_cv
#SBATCH --output=logs/hmm_cv_%j.out
#SBATCH --error=logs/hmm_cv_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --mem=0

set -euo pipefail

cd /path/to/project

module load python
source .venv/bin/activate

mkdir -p logs results/cv

disBatch tasks/hmm_tasks.txt