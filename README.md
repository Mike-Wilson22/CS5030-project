Instructions to run all files on CHPC (assumes CHPC files are copied to a shared folder above data/ when unspecified)

Note: Serial implementation must be run before any parallel implementations in order to validate output correctly.

1. Set up the data
    1. Download data from provided Kaggle website
    2. Extract the file “tracks_features.csv”
    3. Transfer file to CHPC cluster under the new path “data/tracks_features.csv”
2. Transfer utils files (utils.h, utilsGPU.h, utilsGPU.cpp) to CHPC
3. Serial implementation (Transfer serial.cpp to CHPC)
    1. Compile with “g++ serial.cpp -o serial -std=c++17”
    2. Run the generated executable
4. OpenMP implementation (Transfer openMP.cpp to CHPC)
    1. Compile with “g++ openMP.cpp -o openMP -std=c++17”
    2. Run the generated executable
5. GPU (Shared) implementation (Transfer sharedGPU.cu to CHPC)
    1. Load cuda with “module load cuda”
    2. Compile with “nvcc -arch=sm_60 sharedGPU.cu -o sharedGPU”
    3. Run the generated executable
6. MPI implementation (Transfer MPI.cpp to CHPC)
    1. Load mpi with “module load gcc” and “module load intel-mpi”
    2. Compile with “mpicxx MPI.cpp -o mpi -lm”
    3. Run the generated executable with “mpirun -n [number of processes] ./mpi”
7. GPU (Distributed) implementation (Transfer distributedGPUMPI.cpp, distributedGPULaunch.cu, and Makefile to CHPC)
    1. Load the required modules
        1. “module load gcc”
        2. “module load intel-mpi”
        3. “module load cuda”
        4. "module load cmake"
    2. Compile with “make”
    3. Run the generated executable with “mpirun -n [number of processes] ./mpi-cuda”
