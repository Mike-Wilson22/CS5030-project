#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <random>
#include "utils.h"

void kMeans(std::vector<Point>* points, int epochs, int k) {

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

        // Assign all points to initial clusters
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
    
        // Initialize vectors to help with calculating means
        std::vector<int> nPoints;
        std::vector<std::vector<double>> sums;
    
        for (int i = 0; i < k; ++i) {
            nPoints.push_back(0);
        }
        for (int j = 0; j < points->at(0).items.size(); ++j) {
            std::vector<double> sum;
            sums.push_back(sum);
            for (int x = 0; x < k; ++x) {
                sums.at(j).push_back(0.0);
            }
        }

        // Add distance up for each point
        for (int i = 0; i < points->size(); ++i) {
            int clusterId = points->at(i).cluster;
            nPoints[clusterId]++;
            for (int j = 0; j < sums.size(); ++j) {
                sums[j][clusterId] += points->at(i).items.at(j);
            }
    
            points->at(i).minDist = __DBL_MAX__;
        }
    
        // Find mean of all points
        for (int i = 0; i < centroids.size(); ++i) {
            if (nPoints[i] == 0) continue;
            for (int j = 0; j < sums.size(); ++j) {
                centroids.at(i).items.at(j) = sums[j][i] / nPoints[i];
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

    kMeans(&points, 5, 5);

    writeToCSV(&points, "data/output_normalized.csv");
    
    return 0;
}