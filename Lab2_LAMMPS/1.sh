#!/bin/bash -l
#SBATCH -N 1            ##number of used nodes
#SBATCH -n 1            ##number of used cores
##SBATCH --gres=gpu:1   ##if you calculate on gpu tell number of used gpus (this line is commented ##)
#SBATCH -p cpu          ##name of the partition, there is also gpu partition for gpu calculations
##SBATCH -w gn16        ##you can ask for a particular node btw, but you need to know its name (this line is commented ##)

##do necessary module loads here

##run the complied file here

#!/bin/bash
#SBATCH --job-name=a2
#SBATCH --nodes=1
#SBATCH --partition=gpu_small 
#SBATCH --gpus=4
#SBATCH --ntasks=32
#SBATCH --exclusive

PROG="lmp_gpu -sf gpu -pk gpu 4 -in in.surf"

module load apps/lammps-22
module load mpi/openmpi-3.1.2
module load gpu/cuda-9.2
export OMP_NUM_THREADS=4

mpirun -np 32 $PROG

exit 0

