#!/bin/bash
#SBATCH --job-name=improved-fits
#SBATCH --output=logs/improved-fits_%j.out
#SBATCH --error=logs/improved-fits_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=15G
#SBATCH -C genoa



set -euo pipefail

cd /mnt/home/srafilson/code/sequences

module load python
source .venv/bin/activate



disBatch tasks/improved-hmm-fit-tasks.txt