#!/bin/bash
# set the number of nodes and processes per node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=short

# set max wallclock time
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:0
#SBATCH --mem=10GB

# set name of job
#SBATCH --job-name=prac7

# use our reservation
#SBATCH --reservation=cuda202407

module purge
module load CUDA

echo "Running: $1"

$1
