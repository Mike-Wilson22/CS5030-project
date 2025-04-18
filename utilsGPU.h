#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstring>
#include <chrono>
#define ITEM_NUM 11
#define DATA_NUM 1204025
#define START_COL 9
#define END_COL 20

// Define structure Point with all attributes
struct Point {
    double items[ITEM_NUM];    // coordinates
    int cluster;     // no default cluster
    double minDist;  // default infinite dist to nearest cluster
        
    Point(double newItems[ITEM_NUM]) : 
    cluster(-1),
    minDist(__DBL_MAX__) {
            std::memcpy(items, newItems, sizeof(double) * ITEM_NUM);
        }

    Point() :
        items(),
        cluster(-1),
        minDist(0) {}

    double distance(Point *p) {
        double sumDistance = 0;
        for (int i = 0; i < ITEM_NUM; i++) {
            sumDistance += (p->items[i] - items[i]) * (p->items[i] - items[i]);
        }
        return sumDistance;
    }
};

Point* readCSV(std::string filename) {
    std::cout << "Reading file " << std::endl;
    // std::vector<Point> csvVector;
    Point* csvVector = new Point[DATA_NUM];

    std::ifstream csvFile(filename);
    if (!csvFile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return csvVector;
    }

    std::string line;

    // Skip header line if present
    std::getline(csvFile, line);

    int i = 0;
    while (std::getline(csvFile, line)) {
        std::stringstream ss(line);
        std::string token;
        double items[ITEM_NUM];

        int col = 0;
        while (std::getline(ss, token, ',')) {
            if (col >= START_COL && col < END_COL) { // extract columns 9–19
                try {
                    items[col-START_COL] = std::stod(token);
                    ++col;
                } catch (...) {
                    col = START_COL;
                }
            } else {
                ++col;
            }
        }

        csvVector[i] = Point(items);
        i++;
    }

    csvFile.close();
    return csvVector;
}

void writeToCSV(Point* points, std::string filename) {

    std::ofstream myFile;
    myFile.open(filename);

    myFile << "danceability,energy,key,loudness,mode,speechiness,acousticness,instrumentalness,liveness,valence,tempo,cluster" << std::endl;

    for (int i = 0; i < DATA_NUM; i++) {
        double* items = points[i].items;
        for (int j = 0; j < ITEM_NUM; ++j) {
            myFile << items[j] << ",";
        }
        myFile << points[i].cluster << std::endl;
    }
    myFile.close();

    free(points);
}

void compareFiles(std::string filename, std::string filename2) {
    std::ifstream file1(filename);
    std::ifstream file2(filename2);

    if (!file1.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    } else if (!file2.is_open()) {
        std::cerr << "Error opening file: " << filename2 << std::endl;
        return;
    }

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

std::chrono::time_point<std::chrono::high_resolution_clock> startTimerWall() {
    return std::chrono::high_resolution_clock::now();
}

void endTimerWall(std::chrono::time_point<std::chrono::high_resolution_clock> start) {
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken (Wall clock): " << duration.count() << std::endl;
}

std::clock_t startTimerCPU() {
    return std::clock();
}

void endTimerCPU(std::clock_t start) {
    std::clock_t end = std::clock(); 
    double time = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    std::cout << "Time taken (CPU clock): " << time << std::endl;
}

// Normalized read
Point* readCSVNormalized(std::string filename) {
    std::vector<double> stdDevs = {0.18967, 0.29468, 3.5367, 6.982, 0.46968, 0.11599, 0.3852, 0.37628, 0.18046, 0.27048, 30.937};
    std::vector<double> means = {0.493, 0.5095, 5.194, -11.8087, 0.6715, 0.08438, 0.44675, 0.28286, 0.201599, 0.42799, 117.63435};
    Point* csvVector = new Point[DATA_NUM];

    std::ifstream csvFile(filename);
    if (!csvFile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return csvVector;
    }

    std::string line;

    // Skip header line if present
    std::getline(csvFile, line);

    int i = 0;
    while (std::getline(csvFile, line)) {
        std::stringstream ss(line);
        std::string token;
        double items[ITEM_NUM];

        int col = 0;
        while (std::getline(ss, token, ',')) {
            if (col >= START_COL && col < END_COL) { // extract columns 9–19
                try {
                    double newItem = std::stod(token);
                    newItem = (newItem - means[col-START_COL]) / stdDevs[col-START_COL];
                    items[col-START_COL] = newItem;
                    ++col;
                } catch (...) {
                    col = START_COL;
                }
            } else {
                ++col;
            }
        }

        csvVector[i] = Point(items);
        ++i;
    }

    csvFile.close();
    return csvVector;
}