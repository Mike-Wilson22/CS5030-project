#include <string>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <mpi.h>
#include "utilsGPU.h"

void initCuda(double **d_sums, Point **d_points, Point **d_centroids, int **d_nPoints, int **d_k, int k, int **d_size, int localPointsCount, Point *localPoints);
void launchCuda(Point **d_centroids, Point *pointsCentroid, Point **d_points, double *sums, double **d_sums, int **d_nPoints, int *nPoints, int **d_k, int **d_size, int numBlocks, int blockSize);
void getCudaPointsAndFree(Point *localPoints, Point **d_points, int localPointsCount, Point **d_centroids, int **d_k, int **d_size, double **d_sums, int **d_nPoints);
void assignGPU(int rank);


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
    int* pointRecvCounts = new int[size];
    int* pointDispls = new int[size];
    int* clustRecvCounts = new int[size];
    int* clustDispls = new int[size];
    
    // Gather the number of points each process has
    int tempCount = localCount * ITEM_NUM;
    MPI_Gather(&localCount, 1, MPI_INT, clustRecvCounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&tempCount, 1, MPI_INT, pointRecvCounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        pointDispls[0] = 0;
        for (int i = 1; i < size; i++) {
            pointDispls[i] = pointDispls[i-1] + pointRecvCounts[i-1];
        }

        clustDispls[0] = 0;
        for (int i = 1; i < size; i++) {
            clustDispls[i] = clustDispls[i-1] + clustRecvCounts[i-1];
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
                flatAll, pointRecvCounts, pointDispls, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    // Gather clusters
    MPI_Gatherv(localClusters, localCount, MPI_INT,
                allClusters, clustRecvCounts, clustDispls, MPI_INT,
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
    delete[] pointRecvCounts;
    delete[] pointDispls;
    delete[] clustRecvCounts;
    delete[] clustDispls;
    
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
    Point *d_points;
    
    Point *d_centroids;
    
    int *d_nPoints;
    
    int *d_k, *d_size;
    
    int blockSize = 256;
    int numBlocks = (localPointsCount + blockSize - 1) / blockSize;

    initCuda(&d_sums, &d_points, &d_centroids, &d_nPoints, &d_k, k, &d_size, localPointsCount, localPoints);
    
    // Run kmeans algorithm
    for (int x = 0; x < epochs; ++x) {
        int nPoints[K_CLUSTERS] = {0};
        
        // Fill sums array with zeroes
        std::fill(&sums[0][0], &sums[0][0] + ITEM_NUM * K_CLUSTERS, 0.0);

        launchCuda(&d_centroids, pointsCentroid, &d_points, &sums[0][0], &d_sums, &d_nPoints, nPoints, &d_k, &d_size, numBlocks, blockSize);
        
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
    
    // Copy points back to host, free GPU memory
    getCudaPointsAndFree(localPoints, &d_points, localPointsCount, &d_centroids, &d_k, &d_size, &d_sums, &d_nPoints);
    
    // Step 8: Gather results to root
    if (rank == 0) {
        gatherPoints(localPoints, localPointsCount, pointsArray, totalPoints, rank, size);
    } else {
        gatherPoints(localPoints, localPointsCount, nullptr, totalPoints, rank, size);
    }
        
    
    delete[] localPoints;
}


int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    assignGPU(rank);
    
    Point* points = nullptr;
    
    // Only rank 0 reads the CSV file
    if (rank == 0) {
        points = readCSVNormalized("data/tracks_features.csv");
    }

    // for (int i = 0; i < 3; i++) {
    //     std::chrono::time_point<std::chrono::high_resolution_clock> start;
    //     std::clock_t start2;
    //     if (rank == 0) {
    //         std::cout << "Start" << std::endl;
    //         start = startTimerWall();
    //         start2 = startTimerCPU();
    //     }
        
    //     if (rank == 0) {
    //         endTimerWall(start);
    //         endTimerCPU(start2);
    //     }
    // }
    
    std::cout << "[Rank " << rank << "] Started kmeans" << std::endl;
    kMeans(points, 5, K_CLUSTERS, 5);
    std::cout << "[Rank " << rank << "] Finished kmeans" << std::endl;
    
    // Only rank 0 writes the output and compares files
    if (rank == 0) {
        writeToCSV(points, "data/output_gpu_mpi.csv");
        compareFiles("data/output_normalized.csv", "data/output_gpu_mpi.csv");
    }
    
    // Finalize MPI
    MPI_Finalize();
    return 0;
}