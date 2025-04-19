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
#define K_CLUSTERS 5

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

Point* readCSV(std::string filename);

void writeToCSV(Point* points, std::string filename);

void compareFiles(std::string filename, std::string filename2);

std::chrono::time_point<std::chrono::high_resolution_clock> startTimerWall();

void endTimerWall(std::chrono::time_point<std::chrono::high_resolution_clock> start);

std::clock_t startTimerCPU();

void endTimerCPU(std::clock_t start);

// Normalized read
Point* readCSVNormalized(std::string filename);