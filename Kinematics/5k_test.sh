#!/bin/bash

#SBATCH -J gen_kin
#SBATCH -p gpu
#SBATCH -c 8
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -t 00:30:00
#SBATCH --mem=24GB
#SBATCH -o /scratch/users/sydney3/slurm_out/gen_kin_%j.out
#SBATCH -e /scratch/users/sydney3/slurm_out/gen_kin_%j.err
#SBATCH -C GPU_BRD:GEFORCE
#SBATCH -C GPU_MEM:24GB

cd /scratch/users/sydney3
module load py-pytorch/2.0.0_py39
source venvs/skinn_env/bin/activate
module load system ruse
cd /scratch/users/sydney3/forecast/darkenergy-from-LAGN/Kinematics 
ruse --stdout python3 test_5k.py
