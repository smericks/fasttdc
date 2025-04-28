#!/bin/bash -l
#SBATCH -o slurm_%j.out  # STDOUT
#SBATCH -e slurm_%j.err  # STDERR
#SBATCH --time 16:00:00

#SBATCH --ntasks=4       # Total MPI tasks (this will be overridden by launcher.sh)
#SBATCH --cpus-per-task=1 # CPUs per MPI task (this will be overridden by launcher.sh)
####SBATCH --reservation=
####SBATCH --account=

export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python_exec='python'
module load stack/2024-06  gcc/12.2.0 python/3.11.6
if [ "$USE_MPI" = "true" ]; then
    module load openmpi/4.1.6
fi 

# Print model name

echo "arguments: $ARGUMENTS"
echo "option_args: $OPTION_ARGS"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"

# Accessing the environment variable
cpu_per_task=$(($CPU_PER_TASKS))
option_args="${OPTION_ARGS}"

# Compare the value
if [ "$cpu_per_task" -gt 1 ]; then
    option_args="${option_args} --threadCount $cpu_per_task"
    echo "Using $cpu_per_task CPUs per task"
else
    echo "cpu_per_task is 1 or less: $cpu_per_task"
fi

# consruct the directory containing the model
datetime=`date +"%y%m%d-%H%M"`
echo "START : "$datetime

if [ "$USE_MPI" = "true" ]; then
    option_args="${option_args} --use-MPI"
    echo "Running modeling script with arguments: $ARGUMENTS $option_args and with $SLURM_NTASKS MPI tasks"
    mpirun -np $SLURM_NTASKS $python_exec $MODELING_SCRIPT $ARGUMENTS $option_args
else
    echo "Running modeling script without MPI with arguments: $ARGUMENTS $option_args"
    $python_exec $MODELING_SCRIPT $ARGUMENTS $option_args
fi

datetime_end=`date +"%y%m%d-%H%M"`
echo "END : "$datetime_end
