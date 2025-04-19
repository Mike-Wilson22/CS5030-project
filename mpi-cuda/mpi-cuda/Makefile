all:
	mpicc -c mpi-cuda_hello.c -o main.o
	nvcc -c hello_cuda.cu -o cuda_main.o
	mpicc main.o cuda_main.o -lcudart -lstdc++ -llzma -o mpi-cuda 
