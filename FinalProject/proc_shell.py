from mpi4py import MPI

# regular process
class Process:
    def __init__(self, rank) -> None:
        self.rank_ = rank

    def Print(self, msg=""):
        print("CPU", self.rank_, ":",  msg, flush=True)

    def Sync(self, data = None):
        pass

# main process
class Master(Process):
    def __init__(self, rank, size) -> None:
        super().__init__(rank)
        self.size_ = size
        self.Print("Launched " + str(self.size_) +  " processes.")
    
    def Sync(self, data = None):
        # data = comm.bcast(data, root=0)
        pass
        # for i in range(1, self.size_):
        #     self.Print("Broadcas")