#!/bin/sh

#SBATCH --job-name=3-fold-RanForVali
#SBATCH --time=24:00:00
#SBATCH --mail-user=ytpan@umich.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=40

SCRATCH_DIRECTORY=/home/ytpan/slurm/${SLURM_JOBID}
mkdir -p ${SCRATCH_DIRECTORY}
cd ${SCRATCH_DIRECTORY}

source /home/ytpan/.bashrc
source /home/ytpan/.bash_profile

conda activate base

python /home/ytpan/biostat625/final/model_fitting.py
