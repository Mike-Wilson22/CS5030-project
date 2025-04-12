#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

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

// Normalized read
std::vector<Point> readCSVNormalized(std::string filename) {
    std::vector<double> maxes = {0.833, 0.978, 11, 2.225e-308, 1, 0.483, 0.562, 0.34, 0.623, 0.974, 172.848};
    std::vector<double> mins = {0.277, 0.533, 0, -9.563, 0, 0, 0, 0.0247, 0.174, 83.371};
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
                    double newItem = std::stod(token);
                    newItem = (newItem - mins[col-9]) / (maxes[col-9] - mins[col-9]);
                    items.push_back(newItem);
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