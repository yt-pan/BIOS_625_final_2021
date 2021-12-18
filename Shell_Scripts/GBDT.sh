#!/bin/sh

#SBATCH --job-name=GBDT-vali&fit
#SBATCH --time=2-24:00:00
#SBATCH --mail-user=ytpan@umich.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --mem=240G
#SBATCH --cpus-per-task=40

source /home/ytpan/.bashrc
source /home/ytpan/.bash_profile

conda activate base

# python /home/ytpan/biostat625/final/GBDT_model_fitting_test.py
python /home/ytpan/biostat625/final/GBDT_model_fitting.py
