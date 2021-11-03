#!/bin/bash
#SBATCH --job-name=lammps 
#SBATCH --output=res.txt
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100
#SBATCH --error=err.lst

module load compilers/gcc-7.3.1
module load mpi/openmpi-3.1.4
module load apps/lammps-22
module load gpu/cuda-10.2

mpirun lmp_cpu -in charge.in
##mpiexec  lmp_intelcpu  <lj.in

