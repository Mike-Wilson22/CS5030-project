#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <random>
#include <omp.h>
#include <stdexcept>
#include <cuda_runtime.h>
#include "utilsGPU.h"
#define K_CLUSTERS 5

__global__
void assignPoints(Point* points, Point* centroids, int* k, int* length) {
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
            // double dist = centroids[i].distance(p);
            if (dist < p->minDist) {
                p->minDist = dist;
                p->cluster = i;
            }
        }
    }
}

__global__
void sumItems(double* sums, Point* points, int* nPoints, int* length) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < *length) {
        Point* p = &(points[index]);
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

// __global__

void kMeans(std::vector<Point>* points, int epochs, int k, int thread_num) {

    // Initialize centroids
    std::vector<Point> centroids;
    std::vector<int> indices(points->size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine(100));
    
    for (int i = 0; i < k; ++i) {
        centroids.push_back(Point(points->at(indices[i]).items));
    }
    
    if (DATA_NUM != points->size()) {
        std::cerr << "Read " << points->size() << " datapoints, expected " << DATA_NUM << std::endl;
        throw std::runtime_error("Read more or less data than was expected. The CSV is most likely malformed.");
    }
    int size = DATA_NUM;
    Point pointsArray[DATA_NUM];
    std::cout << "size: " << points->size() << std::endl;
    std::copy(points->begin(), points->end(), pointsArray);
    Point* pointsPointer = pointsArray;
    
    Point pointsCentroid[K_CLUSTERS];
    std::copy(centroids.begin(), centroids.end(), pointsCentroid);
    Point* centroidPointer = pointsCentroid;
    
    double sums[ITEM_NUM][K_CLUSTERS] = {0.0};

    int nPoints[k] = {0};
    
    double *d_sums;
    cudaMalloc((void **)&d_sums, ITEM_NUM * K_CLUSTERS * sizeof(double));
    
    Point *d_points;
    cudaMalloc((void **)&d_points, DATA_NUM * sizeof(Point));
    
    Point *d_centroids;
    cudaMalloc((void **)&d_centroids, centroids.size() * sizeof(Point));

    int *d_nPoints;
    cudaMalloc((void **)&d_nPoints, K_CLUSTERS * sizeof(int));
    
    int *d_k, *d_size;
    cudaMalloc((void **)&d_k, sizeof(int));
    cudaMalloc((void **)&d_size, sizeof(int));
    
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    cudaMemcpy(d_k, &k, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_size, &size, sizeof(int), cudaMemcpyHostToDevice);
    // Run kmeans algorithm
    for (int x = 0; x < epochs; ++x) {

        // ------- GPU implementation of first for loop (initialize clusters/distances) ---------
        std::cout << "Calling kernel function 1" << std::endl;

        cudaMemcpy(d_centroids, centroidPointer, centroids.size() * sizeof(Point), cudaMemcpyHostToDevice);
        cudaMemcpy(d_points, pointsPointer, DATA_NUM * sizeof(Point), cudaMemcpyHostToDevice);

        assignPoints<<<numBlocks, blockSize>>>(d_points, d_centroids, d_k, d_size);

        // Just finished kernel call. Check for errors, sync.
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        }

        std::cout << "Kernel function 1 finished" << std::endl;
        
        // Fill sums array with zeroes (replaces second for loop)
        std::fill(&sums[0][0], &sums[0][0] + ITEM_NUM * K_CLUSTERS, 0.0);
        
        
        // ------- GPU implementation of third for loop (initialize clusters/distances) ---------
        std::cout << "copying memory 2" << std::endl;
        cudaMemcpy(d_sums, &sums[0][0], ITEM_NUM * K_CLUSTERS * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nPoints, nPoints, K_CLUSTERS * sizeof(int), cudaMemcpyHostToDevice);
        
        std::cout << "calling kernel function 2" << std::endl;
        
        sumItems<<<numBlocks, blockSize>>>(d_sums, d_points, d_nPoints, d_size);
        
        //Just finished kernel call. Check for errors, sync.
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        }
        
        std::cout << "copying memory back 2" << std::endl;
        cudaMemcpy(&sums[0][0], d_sums, ITEM_NUM * K_CLUSTERS * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(nPoints, d_nPoints, K_CLUSTERS * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Memory is still being edited, so it doesn't need to be copied back until after the last GPU function
        cudaMemcpy(pointsPointer, d_points, DATA_NUM * sizeof(Point), cudaMemcpyDeviceToHost);

        
        // Due to this needing an array per thread, and it only
        // iterating over the points array once, it is not GPU parallelized
        // #pragma omp parallel num_threads(thread_num)
        // {
        //     int nPointsOMP[k] = {0};
        //     #pragma omp for
        //     for (int i = 0; i < points->size(); ++i) {
        //         nPointsOMP[pointsArray[i].cluster]++;
        //     }
            
        //     #pragma omp critical
        //     {
        //         for (int i = 0; i < k; ++i) {
        //             nPoints[i] += nPointsOMP[i];
        //         }
        //     }
        // }
        
        // Find mean of all points
        #pragma omp parallel for num_threads(thread_num)
        for (int i = 0; i < centroids.size(); ++i) {
            if (nPoints[i] == 0) continue;
            for (int j = 0; j < ITEM_NUM; ++j) {
                centroids.at(i).items[j] = sums[j][i] / nPoints[i];
            }
            
        }
    }
    for (int i = 0; i < DATA_NUM; i++) {
        points->at(i) = pointsArray[i];
    }
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_k);
    cudaFree(d_size);
}


int main() {
    
    std::vector<Point> points = readCSV("data/tracks_features.csv");
    if (points.empty()) {
        std::cerr << "Error: No points loaded from CSV." << std::endl;
        return 1;
    }    
    
    kMeans(&points, 5, K_CLUSTERS, 5);
    std::cout << "finished kmeans " << std::endl;
    writeToCSV(&points, "data/output_omp.csv");

    compareFiles("data/output.csv", "data/output_gpu.csv");
    
    return 0;
}