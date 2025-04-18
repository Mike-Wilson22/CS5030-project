#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <random>
#include <omp.h>
#include "utils.h"

void kMeans(std::vector<Point>* points, int epochs, int k, int thread_num) {

    // Initialize centroids
    std::vector<Point> centroids;
    std::vector<int> indices(points->size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine(100));
    
    for (int i = 0; i < k; ++i) {
        centroids.push_back(Point(points->at(indices[i]).items));
    }
    
    // Run kmeans algorithm
    for (int x = 0; x < epochs; ++x) {
        std::vector<std::vector<double>> sums;
        int nPoints[k] = {0};

        # pragma omp parallel num_threads(thread_num)
        {
            int nPointsOMP[k] = {0};

            // Assign all points to initial clusters
            #pragma omp for
            for (int j = 0; j < points->size(); ++j) {
                Point* p = &(points->at(j));
                p->minDist = __DBL_MAX__;  // Reset before checking
                for (int i = 0; i < k; ++i) {
                    double dist = centroids.at(i).distance(p);
                    if (dist < p->minDist) {
                        p->minDist = dist;
                        p->cluster = i;
                    }
                }
            }
                    

            #pragma omp for
            for (int j = 0; j < points->at(0).items.size(); ++j) {
                std::vector<double> sum;
                for (int x = 0; x < k; ++x) {
                    sum.push_back(0.0);
                }
                #pragma omp critical
                {
                    sums.push_back(sum);
                }
            }

            #pragma omp for
            for (int i = 0; i < sums.size(); ++i) {
                for (int j = 0; j < points->size(); ++j) {
                    sums[i][points->at(j).cluster] += points->at(j).items.at(i);
                }
            }
    
            #pragma omp for
            for (int i = 0; i < points->size(); ++i) {
                nPointsOMP[points->at(i).cluster]++;
                points->at(i).minDist = __DBL_MAX__;
            }

            #pragma omp critical
            {
                for (int i = 0; i < k; ++i) {
                    nPoints[i] += nPointsOMP[i];
                }
            }
            
            // Find mean of all points
            #pragma omp barrier
            #pragma omp for
            for (int i = 0; i < centroids.size(); ++i) {
                if (nPoints[i] == 0) continue;
                for (int j = 0; j < sums.size(); ++j) {
                    centroids.at(i).items.at(j) = sums[j][i] / nPoints[i];
                }
                
            }

        }
    }
}


int main() {

    std::vector<Point> points = readCSVNormalized("data/tracks_features.csv");
    if (points.empty()) {
        std::cerr << "Error: No points loaded from CSV." << std::endl;
        return 1;
    }

    // for (int i = 0; i < 3; i++) {
    //     std::cout << "Start" << std::endl;
    //     // auto start = startTimerWall();
    //     auto start = startTimerCPU();
        
        
    //     // endTimerWall(start);
    //     endTimerCPU(start);
    // }
    
    kMeans(&points, 5, 5, 6);

    writeToCSV(&points, "data/output_omp.csv");

    compareFiles("data/output_normalized.csv", "data/output_omp.csv");
    
    return 0;
}

// Time taken to run (wall clock): [0.870513, 0.900094, 0.826877] seconds (16 threads)
// Time taken to run (cpu clock): [10.0785, 10.1108, 10.2722] seconds (16 threads)

// Time taken to run (wall clock): [0.922001, 0.890765, 0.907381] seconds (12 threads)
// Time taken to run (cpu clock): [8.62534, 8.77614, 8.59452] seconds (12 threads)

// Time taken to run (wall clock): [1.213842, 1.165172, 1.190828] seconds (8 threads)
// Time taken to run (cpu clock): [7.35951, 7.39602, 7.31576] seconds (8 threads)

// Time taken to run (wall clock): [1.286735, 1.293697, 1.263102] seconds (7 threads)
// Time taken to run (cpu clock): [7.04313, 7.01414, 7.09844] seconds (7 threads)

// Time taken to run (wall clock): [1.283129, 1.257047, 1.225724] seconds (6 threads)
// Time taken to run (cpu clock): [6.83241, 6.68872, 6.78144] seconds (6 threads)

// Time taken to run (wall clock): [1.442557, 1.446394, 1.465262, 1.510413] seconds (5 threads)
// Time taken to run (cpu clock): [6.86311, 6.77139, 6.86061] seconds (5 threads)

// Time taken to run (wall clock): [1.768315, 1.746593, 1.765074] seconds (4 threads)
// Time taken to run (cpu clock): [6.70337, 6.67672, 6.63795] seconds (4 threads)

// Time taken to run (wall clock): [2.298171 2.299603 2.276771] seconds (3 threads)
// Time taken to run (cpu clock): [6.67818, 6.66136, 6.69502] seconds (3 threads)

// Time taken to run (wall clock): [3.308685, 3.353846, 3.371353] seconds (2 threads)
// Time taken to run (cpu clock): [6.56306, 6.5333, 6.52815] seconds (2 threads)

// Time taken to run (wall clock): [6.514394, 6.601567, 6.590459] seconds (1 thread)
// Time taken to run (cpu clock): [6.58683, 6.55377, 6.57335] seconds (1 thread)