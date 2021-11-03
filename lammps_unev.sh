#!/bin/bash
#SBATCH --job-name=lammps 
#SBATCH --output=res.txt
#SBATCH -N 1 
#SBATCH --ntasks=4 
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100
#SBATCH --error=err.lst

module load compilers/gcc-7.3.1
module load mpi/openmpi-3.1.4
module load apps/lammps-22

mpirun lmp_cpu -in lj_unev.in
##mpiexec -n $NSLOTS  lmp_intelcpu  <lj.in

