#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <random>
#include <omp.h>
#include <cuda_runtime.h>
#include "utilsGPU.h"

__global__
void assignPoints(Point* points, Point* centroids, int* k, int* length) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < *length) {
        Point* p = &(points[index]);
        p->minDist = __DBL_MAX__;  // Reset before checking
        for (int i = 0; i < *k; ++i) {
            double dist = 0;
            Point* c = &(centroids[i]);
            for (int j = 0; j < 11; i++) {
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

void kMeans(std::vector<Point>* points, int epochs, int k, int thread_num) {

    // Initialize centroids
    std::vector<Point> centroids;
    std::vector<int> indices(points->size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine(100));
    
    for (int i = 0; i < k; ++i) {
        centroids.push_back(Point(points->at(indices[i]).items));
    }
    
    int size = DATA_NUM;
    Point pointsArray[DATA_NUM];
    std::cout << "size: " << points->size() << std::endl;
    std::copy(points->begin(), points->end(), pointsArray);
    Point* pointsPointer = pointsArray;

    std::cout << "size: " << centroids.size() << std::endl;
    Point pointsCentroid[5];
    std::copy(centroids.begin(), centroids.end(), pointsCentroid);
    Point* centroidPointer = pointsCentroid;
    
    
    Point *d_points;
    cudaMalloc((void **)&d_points, DATA_NUM * sizeof(Point));

    Point *d_centroids;
    cudaMalloc((void **)&d_centroids, centroids.size() * sizeof(Point));

    int *d_k, *d_size;
    cudaMalloc((void **)&d_k, sizeof(int));
    cudaMalloc((void **)&d_size, sizeof(int));
    

    cudaMemcpy(d_k, &k, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_size, &size, sizeof(int), cudaMemcpyHostToDevice);
    // Run kmeans algorithm
    for (int x = 0; x < epochs; ++x) {

        // Assign all points to initial clusters
        // #pragma omp parallel for num_threads(thread_num)
        // for (int j = 0; j < points->size(); ++j) {
        //     Point* p = &(points->at(j));
        //     p->minDist = __DBL_MAX__;  // Reset before checking
        //     for (int i = 0; i < k; ++i) {
        //         double dist = centroids.at(i).distance(p);
        //         if (dist < p->minDist) {
        //             p->minDist = dist;
        //             p->cluster = i;
        //         }
        //     }
        // }

        // Setup data for GPU

        cudaMemcpy(d_centroids, centroidPointer, centroids.size() * sizeof(Point), cudaMemcpyHostToDevice);
        cudaMemcpy(d_points, pointsPointer, DATA_NUM * sizeof(Point), cudaMemcpyHostToDevice);

        assignPoints<<<ceil(DATA_NUM/256), 256>>>(d_points, d_centroids, d_k, d_size);

        // assignPoints<<<ceil(size/256), 256>>>(d_points, d_centroids, d_k, d_size);
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        assignPoints<<<numBlocks, blockSize>>>(d_points, d_centroids, d_k, d_size);

        //Just finished kernel call. Check for errors, sync.
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        }

        cudaMemcpy(pointsPointer, d_points, DATA_NUM * sizeof(Point), cudaMemcpyDeviceToHost);

        // cudaMemcpy()
        
        // std::copy(points->begin(), points->end(), pointsArray);
        
        
        // Initialize vectors to help with calculating means
        std::vector<std::vector<double>> sums;
        
        for (int j = 0; j < ITEM_NUM; ++j) {
            std::vector<double> sum;
            for (int x = 0; x < k; ++x) {
                sum.push_back(0.0);
            }
            sums.push_back(sum);
        }
        
        #pragma omp parallel for num_threads(thread_num)
        for (int i = 0; i < sums.size(); ++i) {
            for (int j = 0; j < size; ++j) {
                sums[i][pointsArray[j].cluster] += pointsArray[j].items[i];
            }
        }
        
        int nPoints[k] = {0};
        #pragma omp parallel num_threads(thread_num)
        {
            int nPointsOMP[k] = {0};
            #pragma omp for
            for (int i = 0; i < points->size(); ++i) {
                nPointsOMP[pointsArray[i].cluster]++;
                pointsArray[i].minDist = __DBL_MAX__;
            }
            
            #pragma omp critical
            {
                for (int i = 0; i < k; ++i) {
                    nPoints[i] += nPointsOMP[i];
                }
            }
        }
        
        // Find mean of all points
        #pragma omp parallel for num_threads(thread_num)
        for (int i = 0; i < centroids.size(); ++i) {
            if (nPoints[i] == 0) continue;
            for (int j = 0; j < sums.size(); ++j) {
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
    
    kMeans(&points, 5, 5, 5);
    
    writeToCSV(&points, "data/output_omp.csv");

    compareFiles("data/output.csv", "data/output_omp.csv");
    
    return 0;
}