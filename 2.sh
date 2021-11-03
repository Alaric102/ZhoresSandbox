#!/bin/bash -l
#SBATCH -N 1            ##number of used nodes
#SBATCH -n 1            ##number of used cores
##SBATCH --gres=gpu:1   ##if you calculate on gpu tell number of used gpus (this line is commented ##)
#SBATCH -p cpu          ##name of the partition, there is also gpu partition for gpu calculations
##SBATCH -w gn16        ##you can ask for a particular node btw, but you need to know its name (this line is commented ##)

##do necessary module loads here

##run the complied file here

# Don't change anything below this line unless you know what you are doing
module load mpi/openmpi-3.1.4
module load gpu/cuda-9.2
module load apps/lammps-22

lmp_cpu -in lj1.in
##mpiexec -n $NSLOTS  lmp_intelcpu  <lj.in
