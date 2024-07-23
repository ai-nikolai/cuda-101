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
#SBATCH --job-name=prac3

# use our reservation
#SBATCH --reservation=cuda202407

module purge
module load CUDA


# NEW THINGS
sudo -l

echo "using NCU ${NCU}"

cp /home/teaching61/cuda-101/prac3/laplace3d /tmp
cp /home/teaching61/cuda-101/prac3/laplace3d_new /tmp


cd /tmp || exit 1

pwd
ls

sudo -E PATH="${PATH}" LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" ${NCU} --metrics "sm
sp__sass_thread_inst_executed_op_fp32_pred_on.sum,smsp__sass_thread_inst_execute
d_op_integer_pred_on.sum" laplace3d

# sbatch --reservation=cuda202407 --gres=gpu:0 --wrap "sudo -l"