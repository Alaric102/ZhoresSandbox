#!/bin/bash -l
#SBATCH -N 1            ##number of used nodes
#SBATCH -n 4            ##number of used cores
#SBATCH --gres=gpu:1   ##if you calculate on gpu tell number of used gpus (this line is commented ##)
#SBATCH -p gpu          ##name of the partition, there is also gpu partition for gpu calculations
##SBATCH -w gn16        ##you can ask for a particular node btw, but you need to know its name (this line is commented ##)

##do necessary module loads here

##run the complied file here

# Don't change anything below this line unless you know what you are doing
module load mpi/openmpi-3.1.4
module load gpu/cuda-8.0
module load apps/lammps-22

lmp_gpu -sf gpu -pk gpu 1 -in electr.in
##mpiexec -n $NSLOTS  lmp_intelcpu  <lj.in

export OMP_NUM_THREADS=4
