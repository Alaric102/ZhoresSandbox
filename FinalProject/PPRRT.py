# https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/RRT/rrt.py
# mpirun --allow-run-as-root -n 4 python3 FinalProject/PPRRT.py 

# import all the needed modules
from mpi4py import MPI
from proc_shell import Master, Process
from graph import Graph, Node

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

comm.Barrier()

if rank == 0:
    proc = Master(rank, size)
else:
    proc = Process(rank)

