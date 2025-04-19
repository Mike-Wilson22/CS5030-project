#include <vector>
#include <string>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <random>
#include <omp.h>
#include <stdexcept>
#include <cuda_runtime.h>
#include <mpi.h>
#include "utilsGPU.h"
#define K_CLUSTERS 5

__global__
void assignPoints(Point* points, Point* centroids, double* sums, int* nPoints, int* k, int* length) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < *length) {
        Point* p = &(points[index]);
        p->minDist = __DBL_MAX__;  // Reset before checking
        for (int i = 0; i < *k; ++i) {
            double dist = 0;
            Point* c = &(centroids[i]);
            for (int j = 0; j < 11; j++) {
                dist += (c->items[j] - p->items[j]) * (c->items[j] - p->items[j]);
            }

            if (dist < p->minDist) {
                p->minDist = dist;
                p->cluster = i;
            }
        }

        // Technically second for loop
        int cluster = p->cluster;
        for (int i = 0; i < ITEM_NUM; ++i) {
            double value = p->items[i];
            int sumIndex = i * K_CLUSTERS + cluster;
            atomicAdd(&sums[sumIndex], value);
        }
        atomicAdd(&nPoints[cluster], 1);
        p->minDist = __DBL_MAX__;
    }
}

// Helper function to flatten points for MPI communication
double* flattenPoints(Point* points, int numPoints) {
    double* flat = new double[numPoints * ITEM_NUM];
    for (int i = 0; i < numPoints; i++) {
        for (int j = 0; j < ITEM_NUM; j++) {
            flat[i * ITEM_NUM + j] = points[i].items[j];
        }
    }
    return flat;
}

// Helper function to unflatten points after MPI communication
void unflattenPoints(Point* points, double* flat, int numPoints) {
    for (int i = 0; i < numPoints; i++) {
        for (int j = 0; j < ITEM_NUM; j++) {
            points[i].items[j] = flat[i * ITEM_NUM + j];
        }
    }
}

// Helper function to scatter points among MPI processes
Point* scatterPoints(Point* allPoints, int totalPoints, int rank, int size, int& localPointsCount) {
    int pointsPerProc = totalPoints / size;
    int remainder = totalPoints % size;
    
    // Calculate local points count for this process
    localPointsCount = pointsPerProc + (rank < remainder ? 1 : 0);
    
    // Allocate memory for local points
    Point* localPoints = new Point[localPointsCount];
    
    // Calculate send counts and displacements for MPI_Scatterv
    int* sendCounts = new int[size];
    int* displs = new int[size];
    
    for (int i = 0; i < size; i++) {
        sendCounts[i] = (pointsPerProc + (i < remainder ? 1 : 0)) * ITEM_NUM;
        displs[i] = (i == 0) ? 0 : displs[i-1] + sendCounts[i-1];
    }
    
    // Flatten all points for MPI_Scatterv
    double* flatAll = nullptr;
    if (rank == 0) {
        flatAll = flattenPoints(allPoints, totalPoints);
    }
    
    // Allocate memory for local flattened points
    double* flatLocal = new double[localPointsCount * ITEM_NUM];
    
    // Scatter the data
    MPI_Scatterv(flatAll, sendCounts, displs, MPI_DOUBLE,
                 flatLocal, localPointsCount * ITEM_NUM, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    
    // Unflatten local points
    unflattenPoints(localPoints, flatLocal, localPointsCount);
    
    // Clean up
    delete[] flatLocal;
    delete[] sendCounts;
    delete[] displs;
    if (rank == 0) {
        delete[] flatAll;
    }
    
    return localPoints;
}

// Helper function to broadcast centroids to all processes
void broadcastCentroids(Point* centroids, int k) {
    // Flatten centroids for broadcasting
    double* flatCentroids = new double[k * ITEM_NUM];
    
    if (MPI::COMM_WORLD.Get_rank() == 0) {
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < ITEM_NUM; j++) {
                flatCentroids[i * ITEM_NUM + j] = centroids[i].items[j];
            }
        }
    }
    
    // Broadcast flattened centroids
    MPI_Bcast(flatCentroids, k * ITEM_NUM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Unflatten centroids on non-root processes
    if (MPI::COMM_WORLD.Get_rank() != 0) {
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < ITEM_NUM; j++) {
                centroids[i].items[j] = flatCentroids[i * ITEM_NUM + j];
            }
        }
    }
    
    delete[] flatCentroids;
}

// Helper function to reduce cluster sums across all processes
void reduceClusterSums(double* localSums, double* globalSums, int k) {
    MPI_Allreduce(localSums, globalSums, ITEM_NUM * k, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

// Helper function to reduce cluster counts across all processes
void reduceClusterCounts(int* localCounts, int* globalCounts, int k) {
    MPI_Allreduce(localCounts, globalCounts, k, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
}

// Helper function to gather points back to root process
void gatherPoints(Point* localPoints, int localCount, Point* allPoints, int totalPoints, int rank, int size) {
    // Flatten local points for gathering
    double* flatLocal = flattenPoints(localPoints, localCount);
    int* localClusters = new int[localCount];
    
    for (int i = 0; i < localCount; i++) {
        localClusters[i] = localPoints[i].cluster;
    }
    
    // Calculate receive counts and displacements for MPI_Gatherv
    int* recvCounts = new int[size];
    int* displs = new int[size];
    
    // Gather the number of points each process has
    MPI_Gather(&localCount, 1, MPI_INT, recvCounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i-1] + recvCounts[i-1];
        }
    }
    
    // Allocate memory for gathered data on root
    double* flatAll = nullptr;
    int* allClusters = nullptr;
    
    if (rank == 0) {
        flatAll = new double[totalPoints * ITEM_NUM];
        allClusters = new int[totalPoints];
    }
    
    // Gather flattened points
    MPI_Gatherv(flatLocal, localCount * ITEM_NUM, MPI_DOUBLE,
                flatAll, recvCounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    // Gather clusters
    MPI_Gatherv(localClusters, localCount, MPI_INT,
                allClusters, recvCounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);
    
    // Reconstruct points on root
    if (rank == 0) {
        for (int i = 0; i < totalPoints; i++) {
            for (int j = 0; j < ITEM_NUM; j++) {
                allPoints[i].items[j] = flatAll[i * ITEM_NUM + j];
            }
            allPoints[i].cluster = allClusters[i];
        }
    }
    
    // Clean up
    delete[] flatLocal;
    delete[] localClusters;
    delete[] recvCounts;
    delete[] displs;
    
    if (rank == 0) {
        delete[] flatAll;
        delete[] allClusters;
    }
}

void kMeans(Point* pointsArray, int epochs, int k, int thread_num) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int totalPoints = DATA_NUM;
    int localPointsCount;
    
    // Step 1: Scatter data among processes
    Point* localPoints = scatterPoints(pointsArray, totalPoints, rank, size, localPointsCount);
    std::cout << "[Rank " << rank << "] Got " << localPointsCount << " points." << std::endl;
    
    // Step 2: Initialize centroids on root, then broadcast
    Point pointsCentroid[K_CLUSTERS];
    
    if (rank == 0) {
        std::vector<int> indices(totalPoints);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::default_random_engine(100));
        
        for (int i = 0; i < k; ++i) {
            pointsCentroid[i] = Point(pointsArray[indices[i]].items);
        }
    }
    
    broadcastCentroids(pointsCentroid, k);
    
    // Declare arrays for points and centroids
    double sums[ITEM_NUM][K_CLUSTERS] = {0.0};
    
    // Declare all GPU memory structures
    double *d_sums;
    cudaMalloc((void **)&d_sums, ITEM_NUM * K_CLUSTERS * sizeof(double));
    
    Point *d_points;
    cudaMalloc((void **)&d_points, localPointsCount * sizeof(Point));
    
    Point *d_centroids;
    cudaMalloc((void **)&d_centroids, K_CLUSTERS * sizeof(Point));
    
    int *d_nPoints;
    cudaMalloc((void **)&d_nPoints, K_CLUSTERS * sizeof(int));
    
    int *d_k, *d_size;
    cudaMalloc((void **)&d_k, sizeof(int));
    cudaMalloc((void **)&d_size, sizeof(int));
    
    int blockSize = 256;
    int numBlocks = (localPointsCount + blockSize - 1) / blockSize;
    
    cudaMemcpy(d_k, &k, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_size, &localPointsCount, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_points, localPoints, localPointsCount * sizeof(Point), cudaMemcpyHostToDevice);
    
    // Run kmeans algorithm
    for (int x = 0; x < epochs; ++x) {
        int nPoints[K_CLUSTERS] = {0};
        
        // Fill sums array with zeroes
        std::fill(&sums[0][0], &sums[0][0] + ITEM_NUM * K_CLUSTERS, 0.0);
        
        // Update GPU memory structures
        cudaMemcpy(d_centroids, pointsCentroid, K_CLUSTERS * sizeof(Point), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sums, &sums[0][0], ITEM_NUM * K_CLUSTERS * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nPoints, nPoints, K_CLUSTERS * sizeof(int), cudaMemcpyHostToDevice);
        
        assignPoints<<<numBlocks, blockSize>>>(d_points, d_centroids, d_sums, d_nPoints, d_k, d_size);
        
        // Just finished kernel call. Check for errors, sync.
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        }
        
        // Copy needed memory back to host
        cudaMemcpy(&sums[0][0], d_sums, ITEM_NUM * K_CLUSTERS * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(nPoints, d_nPoints, K_CLUSTERS * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Step 5: Reduce to global sums and counts
        double globalSums[ITEM_NUM][K_CLUSTERS] = {0.0};
        int globalCounts[K_CLUSTERS] = {0};
        
        // Flatten sums for MPI_Allreduce
        double* flatLocalSums = new double[ITEM_NUM * K_CLUSTERS];
        for (int i = 0; i < ITEM_NUM; i++) {
            for (int j = 0; j < K_CLUSTERS; j++) {
                flatLocalSums[i * K_CLUSTERS + j] = sums[i][j];
            }
        }
        
        double* flatGlobalSums = new double[ITEM_NUM * K_CLUSTERS];
        
        // Reduce sums and counts
        reduceClusterSums(flatLocalSums, flatGlobalSums, K_CLUSTERS);
        reduceClusterCounts(nPoints, globalCounts, K_CLUSTERS);
        
        // Unflatten global sums
        for (int i = 0; i < ITEM_NUM; i++) {
            for (int j = 0; j < K_CLUSTERS; j++) {
                globalSums[i][j] = flatGlobalSums[i * K_CLUSTERS + j];
            }
        }
        
        // Step 6: Update centroids (All processes do this)
        for (int i = 0; i < K_CLUSTERS; ++i) {
            if (globalCounts[i] == 0) continue;
            for (int j = 0; j < ITEM_NUM; ++j) {
                pointsCentroid[i].items[j] = globalSums[j][i] / globalCounts[i];
            }
        }
        
        // Clean up
        delete[] flatLocalSums;
        delete[] flatGlobalSums;
    }
    
    // Copy points back to host
    cudaMemcpy(localPoints, d_points, localPointsCount * sizeof(Point), cudaMemcpyDeviceToHost);
    
    // Step 8: Gather results to root
    if (rank == 0) {
        gatherPoints(localPoints, localPointsCount, pointsArray, totalPoints, rank, size);
    } else {
        gatherPoints(localPoints, localPointsCount, nullptr, totalPoints, rank, size);
    }
    
    // Free all memory
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_k);
    cudaFree(d_size);
    cudaFree(d_sums);
    cudaFree(d_nPoints);
    
    delete[] localPoints;
}


int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    Point* points = nullptr;
    
    // Only rank 0 reads the CSV file
    if (rank == 0) {
        points = readCSV("data/tracks_features.csv");
    }
    
    std::cout << "[Rank " << rank << "] Started kmeans" << std::endl;
    kMeans(points, 5, K_CLUSTERS, 5);
    std::cout << "[Rank " << rank << "] Finished kmeans" << std::endl;
    
    // Only rank 0 writes the output and compares files
    if (rank == 0) {
        writeToCSV(points, "data/output_gpu_mpi.csv");
        compareFiles("data/output_normalized.csv", "data/output_gpu_mpi.csv");
        free(points);
    }
    
    // Finalize MPI
    MPI_Finalize();
    return 0;
}