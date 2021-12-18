#!/bin/sh

#SBATCH --job-name=plots
#SBATCH --time=2-24:00:00
#SBATCH --mail-user=ytpan@umich.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --mem=350G
#SBATCH --cpus-per-task=1

source /home/ytpan/.bashrc
source /home/ytpan/.bash_profile

conda activate base

python /home/ytpan/biostat625/final/1217/test_set_vis.py