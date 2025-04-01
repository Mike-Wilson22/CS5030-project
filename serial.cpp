#include <omp.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

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

    // Define 2d vector to hold csv, could also be map if needed
    std::vector<Point> csvVector;

    // Open and read file into vector
    std::ifstream csvFile(filename);
    std::string id;
    // Check if there's another line, save id if so
    bool readLabels = false;
    while (std::getline(csvFile, id, ',')) {
        std::vector<double> items;

        // Go through next 6 items
        std::string item;
        for (int i = 1; i < 9; i++) {
            std::getline(csvFile, item, ',');
        }

        // Read values that are being saved currently
        // If the cast fails, there was an additional comma earlier
        // Run the loop from the beginning, try to read the next value
        // If we run into an error, an incorrect value may have been read into the vector
        // Clear out the vector to remove any items that were accidentally read in
        int i = 9;
        while (i < 20) {
            std::getline(csvFile, item, ',');
            if (readLabels) {
                try {
                    items.push_back(std::stod(item));
                } catch (const std::invalid_argument& e) {
                    i = 8;
                    while (items.begin() != items.end()) {
                        items.erase(items.begin());
                    }
                }
            }
            i++;
        }
        if (!readLabels) {readLabels = true;}

        for (int i = 20; i < 24; i++) {
            std::getline(csvFile, item, ',');
        }

        // Last item is delimited by '\n' instead of ','
        std::getline(csvFile, item, '\n');
        if (items.size() > 0) {
            csvVector.push_back(Point(items));
        }
        if (items.size() > 11) {
            int x = 1;
        }
    }

    csvFile.close();

    // Type of item in vector is currently identified by index
    // Could be good to change to a map to identify by name instead
    // Might make splitting data more annoying, however

    return csvVector;
}

void kMeans(std::vector<Point>* points, int epochs, int k) {

    // Initialize centroids
    std::vector<Point> centroids;
    srand(100);
    for (int i = 0; i < k; ++i) {
        std::vector<double> newItems;
        for (int j = 0; j < 11; ++j) {
            int num = rand() % points->size();
            Point p2 = points->at(num);
            double item = p2.items.at(j);
            newItems.push_back(item);
        }
        Point p = Point(newItems);
        centroids.push_back(p);
    }

    // Assign all points to initial clusters
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < points->size(); ++j) {
            Point *p = &(points->at(j));
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

    for (int x = 0; x < epochs; ++x) {
        for (int i = 0; i < points->size(); ++i) {
            int clusterId = points->at(i).cluster;
            nPoints[clusterId]++;
            for (int j = 0; j < sums.size(); ++j) {
                sums[j][clusterId] += points->at(i).items.at(j);
            }
    
            points->at(i).minDist = __DBL_MAX__;
        }
    
        for (int i = 0; i < centroids.size(); ++i) {
            for (int j = 0; j < sums.size(); ++j) {
                centroids.at(i).items.at(j) = sums[i][j] / nPoints[i];
            }
        }
    }
}

void writeToCSV(std::vector<Point>* points, std::string filename) {
    std::cout << "print csv" << std::endl;
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

    kMeans(&points, 5, 5);

    writeToCSV(&points, "data/output.csv");
    
    return 0;
}