#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <random>

// Define structure Point with all attributes
struct Point {
    std::vector<double> items;    // coordinates
    int cluster;     // no default cluster
    double minDist;  // default infinite dist to nearest cluster
        
    Point(std::vector<double> items) : 
        items(items),
        cluster(-1),
        minDist(__DBL_MAX__) {}

    double distance(Point *p) {
        double sumDistance = 0;
        for (int i = 0; i < items.size(); i++) {
            sumDistance += (p->items.at(i) - items.at(i)) * (p->items.at(i) - items.at(i));
        }
        return sumDistance;
    }
};

std::vector<Point> readCSV(std::string filename) {
    std::vector<Point> csvVector;

    std::ifstream csvFile(filename);
    if (!csvFile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return csvVector;
    }

    std::string line;

    // Skip header line if present
    std::getline(csvFile, line);

    while (std::getline(csvFile, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<double> items;

        int col = 0;
        while (std::getline(ss, token, ',')) {
            if (col >= 9 && col < 20) { // extract columns 9–19
                try {
                    items.push_back(std::stod(token));
                } catch (...) {
                    items.clear();
                    break;
                }
            }
            ++col;
        }

        if (!items.empty()) {
            csvVector.push_back(Point(items));
        }
    }

    csvFile.close();
    return csvVector;
}

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

void writeToCSV(std::vector<Point>* points, std::string filename) {

    constexpr int START_COL = 9;
    constexpr int END_COL = 20;

    std::ofstream myFile;
    myFile.open(filename);

    myFile << "danceability,energy,key,loudness,mode,speechiness,acousticness,instrumentalness,liveness,valence,tempo,cluster" << std::endl;

    for (std::vector<Point>::iterator it = points->begin(); it != points->end(); ++it) {
        std::vector<double> items = it->items;
        for (int i = 0; i < items.size(); ++i) {
            myFile << items.at(i) << ",";
        }
        myFile << it->cluster << std::endl;
    }
    myFile.close();
}

int main() {

    std::vector<Point> points = readCSV("data/tracks_features.csv");
    if (points.empty()) {
        std::cerr << "Error: No points loaded from CSV." << std::endl;
        return 1;
    }    

    kMeans(&points, 5, 5);

    writeToCSV(&points, "data/output.csv");
    
    return 0;
}