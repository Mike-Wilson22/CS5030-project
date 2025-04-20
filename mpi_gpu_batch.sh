#!/bin/bash
#SBATCH -n 1 # Number of tasks 
#SBATCH -N 1 # All tasks on one machine 
#SBATCH -p notchpeak-gpu # Partition on some cluster
#SBATCH -A notchpeak-gpu # The account associated with the above partition
#SBATCH -t 00:05:00 # 2 hours (D-HH:MM) 
#SBATCH --gres=gpu:2080ti:4
#SBATCH -o myprog%A%a.out # Standard output 
#SBATCH -e myprog%A%a.err # Standard error

#### IMPORTANT check which account and partition you can use 
#### on the machine you are running on (you can use the 'myallocation' command)

# load modules
module load gcc
module load intel-mpi
module load cuda
module load cmake

#Run the program with our input
# mpirun -n 2 ./mpi-cuda
# mpirun -n 3 ./mpi-cuda
mpirun -n 4 ./mpi-cuda
