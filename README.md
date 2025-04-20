Instructions to run all files on CHPC (assumes CHPC files are copied to a shared folder above data/ when unspecified)

Note: Serial implementation must be run before any parallel implementations in order to validate output correctly.

Set up the data
- Download data from provided Kaggle website
- Extract the file “tracks_features.csv”
- Transfer file to CHPC cluster under the new path “data/tracks_features.csv”
Transfer utils files (utils.h, utilsGPU.h, utilsGPU.cpp) to CHPC
Serial implementation (Transfer serial.cpp to CHPC)
- Compile with “g++ serial.cpp -o serial -std=c++17”
- Run the generated executable
OpenMP implementation (Transfer openMP.cpp to CHPC)
- Compile with “g++ openMP.cpp -o openMP -std=c++17”
- Run the generated executable
GPU (Shared) implementation (Transfer sharedGPU.cu to CHPC)
- Load cuda with “module load cuda”
- Compile with “nvcc -arch=sm_60 sharedGPU.cu -o sharedGPU”
- Run the generated executable
MPI implementation (Transfer MPI.cpp to CHPC)
- Load mpi with “module load gcc” and “module load intel-mpi”
- Compile with “mpicxx MPI.cpp -o mpi -lm”
- Run the generated executable with “mpirun -n [number of processes] ./mpi”
GPU (Distributed) implementation (Transfer distributedGPUMPI.cpp, distributedGPULaunch.cu, and Makefile to CHPC)
- Load the required modules
- “module load gcc”
- “module load intel-mpi”
- “module load cuda”
- "module load cmake"
- Compile with “make”
- Run the generated executable with “mpirun -n [number of processes] ./mpi-cuda”
