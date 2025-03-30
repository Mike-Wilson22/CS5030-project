#include <omp.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

// Define structure Point with all attributes
struct Point {
    std::vector<double> *items;    // coordinates
    int cluster;     // no default cluster
    double minDist;  // default infinite dist to nearest cluster
        
    Point(std::vector<double> *items) : 
        items(items),
        cluster(-1),
        minDist(__DBL_MAX__) {}

    double distance(Point p) {
        double sumDistance;
        for (int i = 0; i < items->size(); i++) {
            sumDistance += (p.items->at(i) - items->at(i)) * (p.items->at(i) - items->at(i));
        }
        return sumDistance;
    }
};


int main() {
    // Define 2d vector to hold csv, could also be map if needed
    std::vector<Point> csvVector;

    // Open and read file into vector
    std::ifstream csvFile("data/tracks_features.csv");
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

        csvVector.push_back(Point(&items));
    }

    csvFile.close();

    // Type of item in vector is currently identified by index
    // Could be good to change to a map to identify by name instead
    // Might make splitting data more annoying, however



    return 0;
}