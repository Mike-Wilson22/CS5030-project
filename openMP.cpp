#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <random>
#include <omp.h>

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

        // Assign all points to initial clusters
        #pragma omp parallel for num_threads(thread_num)
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
        std::vector<std::vector<double>> sums;

        #pragma omp parallel for num_threads(thread_num)
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

        #pragma omp parallel for num_threads(thread_num)
        for (int i = 0; i < sums.size(); ++i) {
            for (int j = 0; j < points->size(); ++j) {
                sums[i][points->at(j).cluster] += points->at(j).items.at(i);
            }
        }

        int nPoints[k] = {0};
        #pragma omp parallel num_threads(thread_num)
        {
            int nPointsOMP[k] = {0};
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
        }
    
        // Find mean of all points
        #pragma omp parallel for num_threads(thread_num)
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

void compareFiles(std::string filename, std::string filename2) {
    std::ifstream file1(filename);
    std::ifstream file2(filename2);

    std::string line1, line2;
    while(getline(file1, line1) && getline(file2, line2)) {
        if (line1 != line2) {
            std::cout << "Files are not the same" <<std::endl;
            return;
        }
    }

    file1.close();
    file2.close();

    std::cout << "Files are the same" <<std::endl;
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