all:
	mpicxx -c distributedGPUMPI.cpp -o main.o
	nvcc -arch=sm_60 -c distributedGPULaunch.cu -o cuda_main.o
	mpicxx -c utilsGPU.cpp -o utilsGPU.o 
	mpicxx main.o cuda_main.o utilsGPU.o -lcudart -lstdc++ -llzma -o mpi-cuda 
