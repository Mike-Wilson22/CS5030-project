#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <mpi.h>
#include <cfloat>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <random>

// Compare output files (validate output)
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

// ------------------- Point Definition -------------------
struct Point {
    std::vector<double> items;
    int cluster;
    double minDist;

    Point(std::vector<double> items) : items(items), cluster(-1), minDist(DBL_MAX) {}

    double distance(const Point& p) const {
        double sum = 0;
        for (size_t i = 0; i < items.size(); ++i) {
            sum += (p.items[i] - items[i]) * (p.items[i] - items[i]);
        }
        return sum;
    }
};

//Helper functions for MPI


std::vector<Point> initializeRandomCentroids(const std::vector<Point>& points, int k) {
    std::vector<int> indices(points.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine(100));

    std::vector<Point> centroids;
    for (int i = 0; i < k; ++i) {
        centroids.push_back(Point(points[indices[i]].items));
    }
    return centroids;
}



// ------------------- Broadcast Centroids to All Processes -------------------
void broadcastCentroids(std::vector<Point>& centroids) {
    int k, dim;

    if (MPI::COMM_WORLD.Get_rank() == 0) {
        k = centroids.size();
        dim = centroids[0].items.size();
    }

    // Broadcast k and dim to all ranks
    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<double> flatCentroids;
    if (MPI::COMM_WORLD.Get_rank() == 0) {
        for (const Point& c : centroids)
            flatCentroids.insert(flatCentroids.end(), c.items.begin(), c.items.end());
    }

    flatCentroids.resize(k * dim);
    MPI_Bcast(flatCentroids.data(), k * dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Reconstruct centroids on other ranks
    if (MPI::COMM_WORLD.Get_rank() != 0) {
        centroids.clear();
        for (int i = 0; i < k; ++i) {
            std::vector<double> items(flatCentroids.begin() + i * dim, flatCentroids.begin() + (i + 1) * dim);
            centroids.emplace_back(items);
        }
    }
}


//MPI_Scatter can't use a vector of Points. It needs doubles.
std::vector<double> flattenPoints(const std::vector<Point>& points) {
    std::vector<double> flat;
    for (const Point& p : points) {
        flat.insert(flat.end(), p.items.begin(), p.items.end());
    }
    return flat;
}

//Similar logic for the clusters.
std::vector<int> flattenClusters(const std::vector<Point>& points) {
    std::vector<int> flat;
    for (const Point& p : points) {
        flat.push_back(p.cluster);
    }
    return flat;
}


//Reconstruct the Points from the flattened versions.
std::vector<Point> unflattenPoints(const std::vector<double>& flat, int dim) {
    std::vector<Point> result;
    for (size_t i = 0; i < flat.size(); i += dim) {
        std::vector<double> items(flat.begin() + i, flat.begin() + i + dim);
        result.emplace_back(items);
    }
    return result;
}

std::vector<Point> scatterPoints(std::vector<Point>& allPoints, int rank, int size) {
    int dim;
    if (rank == 0) {
        dim = allPoints[0].items.size();
    }
    // Broadcast dimension size from root to everyone
    MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int totalPoints;
    if (rank == 0) {
        totalPoints = allPoints.size();
    }
    MPI_Bcast(&totalPoints, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int pointsPerProc = totalPoints / size;
    int remainder = totalPoints % size;

    std::vector<int> sendCounts(size), displs(size);
    for (int i = 0; i < size; ++i) {
        sendCounts[i] = (pointsPerProc + (i < remainder ? 1 : 0)) * dim;
        displs[i] = (i == 0) ? 0 : displs[i - 1] + sendCounts[i - 1];
    }

    std::vector<double> flatAll;
    if (rank == 0) {
        flatAll = flattenPoints(allPoints);
    }

    int localCount = sendCounts[rank];
    std::vector<double> flatLocal(localCount);

    MPI_Scatterv(flatAll.data(), sendCounts.data(), displs.data(), MPI_DOUBLE,
                 flatLocal.data(), localCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return unflattenPoints(flatLocal, dim);
}


std::vector<Point> gatherPoints(std::vector<Point>& localPoints, int rank, int size) {
    int dim = localPoints[0].items.size();
    int localCount = localPoints.size();

    std::vector<double> flatLocalItems = flattenPoints(localPoints);
    std::vector<int> flatLocalClusters = flattenClusters(localPoints);

    std::vector<int> recvCounts(size), displs(size);
    int totalLocalDoubles = localCount * dim;
    int totalLocalClusters = localCount;

    MPI_Gather(&totalLocalDoubles, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<double> flatAllItems;
    std::vector<int> flatAllClusters;
    if (rank == 0) {
        int total = 0;
        for (int i = 0; i < size; ++i) {
            displs[i] = total;
            total += recvCounts[i];
        }
        flatAllItems.resize(total);
        flatAllClusters.resize(total / dim);  // one cluster per point
    }

    MPI_Gatherv(flatLocalItems.data(), totalLocalDoubles, MPI_DOUBLE,
                flatAllItems.data(), recvCounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    std::vector<int> clusterRecvCounts(size);
    for (int i = 0; i < size; ++i) {
        clusterRecvCounts[i] = recvCounts[i] / dim;  // one int per point
    }

    std::vector<int> clusterDispls(size);
    clusterDispls[0] = 0;
    for (int i = 1; i < size; ++i) {
        clusterDispls[i] = clusterDispls[i - 1] + clusterRecvCounts[i - 1];
    }

    MPI_Gatherv(flatLocalClusters.data(), totalLocalClusters, MPI_INT,
                flatAllClusters.data(), clusterRecvCounts.data(), clusterDispls.data(), MPI_INT,
                0, MPI_COMM_WORLD);

    // Reconstruct full Points
    if (rank == 0) {
        std::vector<Point> result;
        for (size_t i = 0; i < flatAllClusters.size(); ++i) {
            std::vector<double> items(flatAllItems.begin() + i * dim, flatAllItems.begin() + (i + 1) * dim);
            Point p(items);
            p.cluster = flatAllClusters[i];
            result.push_back(p);
        }
        return result;
    } else {
        return std::vector<Point>();
    }
}


std::vector<std::vector<double>> reduceClusterSums(std::vector<std::vector<double>>& localSums, int k, int rank, int size) {
    int dim = localSums[0].size();
    std::vector<double> localFlat, globalFlat(k * dim);

    for (const auto& vec : localSums)
        localFlat.insert(localFlat.end(), vec.begin(), vec.end());

    MPI_Allreduce(localFlat.data(), globalFlat.data(), k * dim, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    std::vector<std::vector<double>> result(k, std::vector<double>(dim));
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < dim; ++j)
            result[i][j] = globalFlat[i * dim + j];

    return result;
}

std::vector<int> reduceClusterCounts(const std::vector<int>& localCounts, int k, int rank, int size) {
    std::vector<int> globalCounts(k);
    MPI_Allreduce(localCounts.data(), globalCounts.data(), k, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return globalCounts;
}

// ------------------- Read CSV (Only rank 0 reads in MPI) -------------------
std::vector<Point> readCSV(std::string filename) {
    std::vector<Point> csvVector;

    std::ifstream csvFile(filename);
    if (!csvFile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return csvVector;
    }

    std::string line;
    std::getline(csvFile, line); // skip header

    while (std::getline(csvFile, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<double> items;

        int col = 0;
        while (std::getline(ss, token, ',')) {
            if (col >= 9 && col < 20) {
                try {
                    items.push_back(std::stod(token));
                    ++col;
                } catch (...) {
                    col = 9;
                }
            } else {
                ++col;
            }
        }

        if (!items.empty()) {
            csvVector.emplace_back(items);
        }
    }

    csvFile.close();
    return csvVector;
}

// Just rank 0 needs to do this
void writeToCSV(const std::vector<Point>& points, const std::string& filename){
    std::cout << "print csv" << std::endl;
    std::ofstream myFile;
    myFile.open(filename);

    myFile << "danceability,energy,key,loudness,mode,speechiness,acousticness,instrumentalness,liveness,valence,tempo,cluster" << std::endl;
    for (const Point& point : points) {
        for (size_t i = 0; i < point.items.size(); ++i) {
            myFile << point.items[i] << ",";
        }
        myFile << point.cluster << "\n";
    }
    
    myFile.close();
}

// ------------------- Core KMeans with MPI -------------------
void kMeansMPI(std::vector<Point>& allPoints, int k, int epochs, int rank, int size) {
    // Step 1: Scatter data among processes (Split allPoints)
    std::vector<Point> localPoints = scatterPoints(allPoints, rank, size); 
    std::cout << "[Rank " << rank << "] Got " << localPoints.size() << " points.\n";


    // Step 2: Initialize centroids on root, then broadcast
    std::vector<Point> centroids;
    if (rank == 0) {
        centroids = initializeRandomCentroids(allPoints, k);
    }
    broadcastCentroids(centroids);
    if (rank != 0) {
        std::cout << "[Rank " << rank << "] Got " << centroids.size() << " centroids with dim = " 
                  << centroids[0].items.size() << std::endl;
    }                      

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Step 3: Assign clusters (Same logic as OpenMP, but just on localPoints)
        for (Point& p : localPoints) {
            p.minDist = DBL_MAX;  // Reset BEFORE assigning
            for (int i = 0; i < k; ++i) {
                double dist = centroids[i].distance(p);
                if (dist < p.minDist) {
                    p.minDist = dist;
                    p.cluster = i;
                }
            }
        }

        // Step 4: Local sums and counts
        size_t dim = (localPoints.empty()) ? 11 : localPoints[0].items.size(); 

        std::vector<std::vector<double>> localSums(k, std::vector<double>(dim, 0.0));
        std::vector<int> localCounts(k, 0);
        
        for (const Point& p : localPoints) {
            localCounts[p.cluster]++;
            for (size_t i = 0; i < p.items.size(); ++i) {
                localSums[p.cluster][i] += p.items[i];
            }
        }

        // Step 5: Reduce to global sums and counts
        std::vector<std::vector<double>> globalSums = reduceClusterSums(localSums, k, rank, size);
        std::vector<int> globalCounts = reduceClusterCounts(localCounts, k, rank, size);

        // Step 6: Update centroids (All processes do this)
        for (int i = 0; i < k; ++i) {
            for (size_t j = 0; j < centroids[i].items.size(); ++j) {
                centroids[i].items[j] = (globalCounts[i] > 0) ? (globalSums[i][j] / globalCounts[i]) : centroids[i].items[j];
            }
        }

        // Step 7: Reset distances for next iteration
        for (Point& p : localPoints) {
            p.minDist = DBL_MAX;
        }
    }

    // Step 8: Gather results to root
    std::vector<Point> allClusteredPoints = gatherPoints(localPoints, rank, size);


    if (rank == 0) {
        std::map<int, int> clusterCounts;
        for (const Point& pt : allClusteredPoints) {
            clusterCounts[pt.cluster]++;
        }
        std::cout << "Cluster distribution:\n";
        for (const auto& [c, count] : clusterCounts) {
            std::cout << "  Cluster " << c << ": " << count << " points\n";
        }
    }
    
    // Step 9: Only root writes to CSV
    if (rank == 0) {
        writeToCSV(allClusteredPoints, "data/output_mpi.csv");
    }
    if (rank == 0) {
        std::cout << "Wrote clustered data to output_mpi.csv\n";
        compareFiles("data/output.csv", "data/output_mpi.csv");
    }
    
}



int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<Point> allPoints;
    if (rank == 0) {
        allPoints = readCSV("data/tracks_features.csv");
    }

    kMeansMPI(allPoints, 5, 5, rank, size);

    MPI_Finalize();
    return 0;
}


