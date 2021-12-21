from mpi4py import MPI

# regular process
class Process:
    def __init__(self, rank) -> None:
        self.rank_ = rank

    def print(self, msg):
        print(self.rank_, "report:", msg)

# main process
class Master(Process):
    def __init__(self, rank, size) -> None:
        super().__init__(rank)
        self.size_ = size
        self.print("Launched ", self.size_, "processes.")