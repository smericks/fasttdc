#!/usr/bin/env bash

#!/bin/bash -l

queue_name="normal.24h"
export MODELING_SCRIPT='step07_test_mpi.py'
export USE_MPI='false'

# Define the number of MPI tasks and the number of threads per task
if [ "$USE_MPI" = "true" ]; then
    mpi_tasks=4
    cpus_per_task=1  # number of threads (?)
else
    mpi_tasks=1
    cpus_per_task=1
fi
export CPU_PER_TASKS="$cpus_per_task"

start_file='start.slurm'

# Define the arguments as a list
arguments_list=("")
optionlist=("")
job_name='TDC_sampling'

# Loop over the list and submit a job for each argument
for ((i = 0; i < ${#arguments_list[@]}; i++)); do
    ARGUMENT="${arguments_list[i]}"
    OPTION_ARGS="${optionlist[i]}"

    export ARGUMENTS="$ARGUMENT"
    export OPTION_ARGS="$OPTION_ARGS"

    sbatch --ntasks="$mpi_tasks" --cpus-per-task="$cpus_per_task" -J "${job_name}_${ARGUMENT// /_}" -p "$queue_name" --mem-per-cpu=8G "$start_file"
    echo "Job submitted for argument: $ARGUMENT with $mpi_tasks MPI tasks and $cpus_per_task CPUs per task."

    sleep 2
done

