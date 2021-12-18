#!/bin/sh

#SBATCH --job-name=RF_Vali&fit
#SBATCH --time=3-24:00:00
#SBATCH --mail-user=ytpan@umich.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --mem=340G
#SBATCH --cpus-per-task=28

source /home/ytpan/.bashrc
source /home/ytpan/.bash_profile

conda activate base

#python /home/ytpan/biostat625/final/RF_model_fitting_test.py
python /home/ytpan/biostat625/final/RF_model_fitting.py
