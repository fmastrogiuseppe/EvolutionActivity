#!/bin/bash
#SBATCH --mail-user=f.mastrogiuseppe@ucl.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --job-name=IntegralSingle
#SBATCH --output=monitor.out
#SBATCH --time=30-00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=1G
#SBATCH --partition=cpu

python -u compute_integrals.py ${SLURM_ARRAY_TASK_ID}