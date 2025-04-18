#include <vector>
#include <string>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <random>
#include <omp.h>
#include <stdexcept>
#include <cuda_runtime.h>
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


void kMeans(Point* pointsArray, int epochs, int k, int thread_num) {

    // Initialize centroids
    Point pointsCentroid[K_CLUSTERS];

    std::vector<int> indices(DATA_NUM);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine(100));
    
    for (int i = 0; i < k; ++i) {
        pointsCentroid[i] = Point(pointsArray[indices[i]].items);
    }

    // Declare arrays for points and centroids
    int size = DATA_NUM;

    double sums[ITEM_NUM][K_CLUSTERS] = {0.0};

    // Declare all GPU memory structures
    double *d_sums;
    cudaMalloc((void **)&d_sums, ITEM_NUM * K_CLUSTERS * sizeof(double));
    
    Point *d_points;
    cudaMalloc((void **)&d_points, DATA_NUM * sizeof(Point));
    
    Point *d_centroids;
    cudaMalloc((void **)&d_centroids, K_CLUSTERS * sizeof(Point));
    
    int *d_nPoints;
    cudaMalloc((void **)&d_nPoints, K_CLUSTERS * sizeof(int));
    
    int *d_k, *d_size;
    cudaMalloc((void **)&d_k, sizeof(int));
    cudaMalloc((void **)&d_size, sizeof(int));
    
    int blockSize = 16;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    cudaMemcpy(d_k, &k, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_size, &size, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_points, pointsArray, DATA_NUM * sizeof(Point), cudaMemcpyHostToDevice);
    
    // Run kmeans algorithm
    for (int x = 0; x < epochs; ++x) {

        int nPoints[K_CLUSTERS] = {0};
        
        // Fill sums array with zeroes (replaces second for loop)
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
                
        // Find mean of all points
        #pragma omp parallel for num_threads(thread_num)
        for (int i = 0; i < K_CLUSTERS; ++i) {
            if (nPoints[i] == 0) continue;
            for (int j = 0; j < ITEM_NUM; ++j) {
                pointsCentroid[i].items[j] = sums[j][i] / nPoints[i];
            }
            
        }
        
    }

    // Copy points back to host, copy array back into vector
    cudaMemcpy(pointsArray, d_points, DATA_NUM * sizeof(Point), cudaMemcpyDeviceToHost);

    // Free all memory
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_k);
    cudaFree(d_size);
    cudaFree(d_sums);
    cudaFree(d_nPoints);
}


int main() {
    
    Point* points = readCSVNormalized("data/tracks_features.csv");
        
    // for (int i = 0; i < 3; i++) {
    //     std::cout << "Start" << std::endl;
    //     auto start2 = startTimerCPU();
    //     auto start = startTimerWall();
        
        
    //     endTimerWall(start);
    //     endTimerCPU(start2);
    // }

    kMeans(points, 5, K_CLUSTERS, 5);
    
    writeToCSV(points, "data/output_gpu.csv");

    compareFiles("data/output_normalized.csv", "data/output_gpu.csv");
    
    return 0;
}

// First execution is notably slower then others--this may be due to memory loading onto GPU?

// Time taken to run (wall clock): [1.587205, 1.824759, 1.607862] seconds (block size 512 threads)
// Time taken to run (cpu clock): [1.58146, 1.59184, 1.59481] seconds (block size 512 threads)

// Time taken to run (wall clock): [1.613788, 1.631238, 1.620053] seconds (block size 256 threads)
// Time taken to run (cpu clock): [1.60671, 1.62136, 1.60614] seconds (block size 256 threads)

// Time taken to run (wall clock): [1.603406, 1.630459, 1.605275] seconds (block size 128 threads)
// Time taken to run (cpu clock): [1.59254, 1.62097, 1.59973] seconds (block size 128 threads)

// Time taken to run (wall clock): [1.607722, 1.459259, 1.469630] seconds (block size 64 threads)
// Time taken to run (cpu clock): [1.56956, 1.45311, 1.46486] seconds (block size 64 threads)

// Time taken to run (wall clock): [0.536088, 0.530974, 0.531105] seconds (block size 32 threads)
// Time taken to run (cpu clock): [0.532402, 0.527215, 0.526539] seconds (block size 32 threads)

// Time taken to run (wall clock): [0.719224, 0.721331, 0.711376] seconds (block size 16 threads)
// Time taken to run (cpu clock): [0.715885, 0.71655, 0.709268] seconds (block size 16 threads)