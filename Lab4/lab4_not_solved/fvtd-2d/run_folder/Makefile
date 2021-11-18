MPICC = mpicxx

LIB  = -lgomp -lm

GCC_COPT = -fopenmp

VPATH = ../advection:../

SOURCES = transport.cpp main_2d_mpi.cpp
OBJECTS = $(SOURCES:.cpp=_cpp.o)

all: 2D_MPI_CUDA.exe

2D_MPI_CUDA.exe: $(OBJECTS) $(CUDA_OBJECTS)
	$(MPICC) $^ -o $@ $(LIB)

%_cpp.o: %.cpp
	$(MPICC) -c $< -o $@ $(GCC_COPT)

1_p:
	cp ../../1_p/*txt ./

2_p:
	cp ../../2_p/*txt ./

4_p:
	cp ../../4_p/*txt ./

8_p:
	cp ../../8_p/*txt ./

clean:
	rm -rf *.o
	rm -rf *.exe
	rm -rf *.txt
	rm -rf *.png
