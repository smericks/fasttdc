#!/bin/bash

#SBATCH -J gen_kin
#SBATCH -p kipac,normal,hns
#SBATCH -c 4
#SBATCH -N 1
#SBATCH -t 00:90:00
#SBATCH --mem 8000
#SBATCH --error /scratch/users/sydney3/slurm_out/gen_kin_%j.err
#SBATCH --output /scratch/users/sydney3/slurm_out/gen_kin_%j.out

cd /scratch/users/sydney3
module load python/3.9
source venvs/forecast_env/bin/activate
cd forecast/darkenergy-from-LAGN/Modeling/Kinematics
# first argument passed to sbatch is the lens idx!
python3 galkin_ifu_sherlock_script.py --lens_idx=$1 --h5_posteriors_path=/scratch/users/sydney3/forecast/darkenergy-from-LAGN/Experiments/lsst_forecast/DataVectors/gold/quad_posteriors_KIN.h5
