#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "utilsGPU.h"
#include <cuda_runtime.h>

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

extern "C++" void initCuda(double **d_sums, Point **d_points, Point **d_centroids, int **d_nPoints, int **d_k, int k, int **d_size, int localPointsCount, Point *localPoints) {
    // Declare all GPU memory structures
    cudaMalloc((void **)d_sums, ITEM_NUM * K_CLUSTERS * sizeof(double));
    
    cudaMalloc((void **)d_points, localPointsCount * sizeof(Point));
    
    cudaMalloc((void **)d_centroids, K_CLUSTERS * sizeof(Point));
    
    cudaMalloc((void **)d_nPoints, K_CLUSTERS * sizeof(int));
    
    cudaMalloc((void **)d_k, sizeof(int));
    cudaMalloc((void **)d_size, sizeof(int));
    
    cudaMemcpy(*d_k, &k, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_size, &localPointsCount, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_points, localPoints, localPointsCount * sizeof(Point), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error in initCuda: " << cudaGetErrorString(err) << std::endl;
    }

}

extern "C++" void launchCuda(Point **d_centroids, Point *pointsCentroid, Point **d_points, double *sums, double **d_sums, int **d_nPoints, int *nPoints, int **d_k, int **d_size, int numBlocks, int blockSize) {
    cudaMemcpy(*d_centroids, pointsCentroid, K_CLUSTERS * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_sums, sums, ITEM_NUM * K_CLUSTERS * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_nPoints, nPoints, K_CLUSTERS * sizeof(int), cudaMemcpyHostToDevice);

    int checkSize = -1;
    cudaMemcpy(&checkSize, *d_size, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Size is: " << checkSize << std::endl;

    assignPoints<<<numBlocks, blockSize>>>(*d_points, *d_centroids, *d_sums, *d_nPoints, *d_k, *d_size);
    
    // Just finished kernel call. Check for errors, sync.
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error in launchCuda before function: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Copy needed memory back to host
    cudaMemcpy(sums, *d_sums, ITEM_NUM * K_CLUSTERS * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(nPoints, *d_nPoints, K_CLUSTERS * sizeof(int), cudaMemcpyDeviceToHost);
}

extern "C++" void getCudaPointsAndFree(Point *localPoints, Point **d_points, int localPointsCount, Point **d_centroids, int **d_k, int **d_size, double **d_sums, int **d_nPoints) {
    
    cudaMemcpy(localPoints, *d_points, localPointsCount * sizeof(Point), cudaMemcpyDeviceToHost);

    cudaFree(*d_points);
    cudaFree(*d_centroids);
    cudaFree(*d_k);
    cudaFree(*d_size);
    cudaFree(*d_sums);
    cudaFree(*d_nPoints);
}

extern "C++" void assignGPU(int rank) {
    cudaSetDevice(rank);
}