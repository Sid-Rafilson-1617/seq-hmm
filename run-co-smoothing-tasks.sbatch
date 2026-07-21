#!/bin/bash
#SBATCH --job-name=co-smooth-full
#SBATCH --output=logs/co-smooth-full_%j.out
#SBATCH --error=logs/co-smooth-full_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=15G
#SBATCH -C genoa



set -euo pipefail

cd /mnt/home/srafilson/code/sequences

module load python
source .venv/bin/activate

mkdir -p logs results/cv



disBatch tasks/co-smoothing-tasks.txt