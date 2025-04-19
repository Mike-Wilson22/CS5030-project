cmake_minimum_required(VERSION 3.18)

# Project Name
project(MPI_CUDA)

# Set CUDA as required language
find_package(CUDAToolkit REQUIRED)

# Find MPI
find_package(MPI REQUIRED)


# Define the source files
set(SOURCE_FILES
    distributedGPUMPI.cpp
    distributedGPULaunch.cu  
)

enable_language(CUDA)

add_library(kernels
  distributedGPULaunch.cu
)

target_compile_features(kernels PUBLIC cxx_std_17)
set_target_properties(kernels
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(kernels)


add_executable(mpi-cuda distributedGPUMPI.cpp)

# Link MPI and CUDA libraries
target_link_libraries(mpi-cuda PRIVATE
        kernels
        MPI::MPI_CXX
	CUDA::cudart
	)

# Optional: For more fine-grained control over CUDA settings, you can specify additional flags
set_target_properties(mpi-cuda PROPERTIES
    CUDA_ARCHITECTURES "60"  # Set the target architecture for CUDA (adjust as needed)
)
