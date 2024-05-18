#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>

using namespace std;

#define INF 1e9

void readInput(const char* filename, int total_nodes, int* residual) {
    ifstream file;
    file.open(filename);

    if (!file) {
        cout << "Error reading file!" << endl;
        exit(1);
    }

    string line;
    int source, destination;
    int numberOfEdges = 0;
    float capacity;

    while (getline(file, line)) {
        if (line.empty()) continue;

        stringstream linestream(line);
        if (!(linestream >> source >> destination >> capacity)) {
            cerr << "Error parsing line: " << line << endl;
            continue;
        }

        source--;
        destination--;
        int scaledCapacity = static_cast<int>(capacity * 1000);

        if (!residual) {
            cerr << "Memory allocation failed for residual matrix." << endl;
            exit(EXIT_FAILURE);
        }

        numberOfEdges++;
        residual[source * total_nodes + destination] = scaledCapacity;
    }

    cout << "Number of edges in graph is: " << numberOfEdges << endl;
    file.close();
}

__global__ void bfsTD(int *r_capacity, int *parent, int *flow, bool *frontier, bool *visited, int vertices, int sink, int* locks) {
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (!frontier[sink] && Idx < vertices && frontier[Idx]) {
        frontier[Idx] = false;
        visited[Idx] = true;

        for (int i = vertices; i > Idx; i--) {
            if (!frontier[i] && !visited[i] && r_capacity[Idx * vertices + i] > 0) {
                if (atomicCAS(locks + i, 0, 1) == 1 || frontier[i]) {
                    continue;
                }
                frontier[i] = true;
                locks[i] = 0;
                parent[i] = Idx;
                flow[i] = min(flow[Idx], r_capacity[Idx * vertices + i]);
            }
        }

        for (int i = 0; i < Idx; i++) {
            if (!frontier[i] && !visited[i] && r_capacity[Idx * vertices + i] > 0) {
                if (atomicCAS(locks + i, 0, 1) == 1 || frontier[i]) {
                    continue;
                }
                frontier[i] = true;
                locks[i] = 0;
                parent[i] = Idx;
                flow[i] = min(flow[Idx], r_capacity[Idx * vertices + i]);
            }
        }
    }
}

__global__ void bfsBU(int *r_capacity, int *parent, int *flow, bool *frontier, bool *visited, int vertices, int source, int* locks) {
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (!frontier[source] && Idx < vertices && frontier[Idx]) {
        frontier[Idx] = false;
        visited[Idx] = true;

        for (int i = 0; i < Idx; i++) {
            if (!frontier[i] && !visited[i] && r_capacity[i * vertices + Idx] > 0) {
                if (atomicCAS(locks + i, 0, 1) == 1 || frontier[i]) {
                    continue;
                }
                frontier[i] = true;
                locks[i] = 0;
                parent[i] = Idx;
                flow[i] = min(flow[Idx], r_capacity[i * vertices + Idx]);
            }
        }

        for (int i = vertices; i > Idx; i--) {
            if (!frontier[i] && !visited[i] && r_capacity[i * vertices + Idx] > 0) {
                if (atomicCAS(locks + i, 0, 1) == 1 || frontier[i]) {
                    continue;
                }
                frontier[i] = true;
                locks[i] = 0;
                parent[i] = Idx;
                flow[i] = min(flow[Idx], r_capacity[i * vertices + Idx]);
            }
        }
    }
}

__global__ void cudaAugment_path(int* parent, bool* do_change_capacity, int total_nodes, int* r_capacity, int path_flow) {
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (Idx < total_nodes && do_change_capacity[Idx]) {
        r_capacity[parent[Idx] * total_nodes + Idx] -= path_flow;
        r_capacity[Idx * total_nodes + parent[Idx]] += path_flow;
    }
}

bool sink_reachable(bool* frontier, int total_nodes, int sink) {
    for (int i = total_nodes - 1; i > -1; --i) {
        if (frontier[i]) {
            return i == sink;
        }
    }
    return false;
}

float edmondskarp(const char* filename, int total_nodes) {
    cudaEvent_t startEvent3, stopEvent3, startEvent3_1, stopEvent3_1;
    cudaEventCreate(&startEvent3);
    cudaEventCreate(&stopEvent3);
    cudaEventCreate(&startEvent3_1);
    cudaEventCreate(&stopEvent3_1);
    float partinitmili = 0.0f;
    float initmili = 0.0f;
    float totalInitTime = 0.0f;
    cudaEventRecord(startEvent3);

    int* residual;

    float avgBFSTime = 0;
    int bfsCounter = 0;
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEvent_t start, stop;
    float milliseconds = 0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    float avgAUGTime = 0;
    int augCounter = 0;
    cudaEvent_t startEvent2, stopEvent2;
    cudaEventCreate(&startEvent2);
    cudaEventCreate(&stopEvent2);

    try {
        residual = new int[total_nodes * total_nodes]();
    } catch (const std::bad_alloc& e) {
        std::cerr << "Failed to allocate memory for the residual matrix: " << e.what() << std::endl;
        return 1;
    }

    readInput(filename, total_nodes, residual);
    cout << "data read" << endl;

    int source = 0;
    int sink = total_nodes - 1;
    int path_flow;

    int* parent = new int[total_nodes];
    int* flow = new int[total_nodes];
    bool* frontier = new bool[total_nodes];
    bool* visited = new bool[total_nodes];
    bool* do_change_capacity = new bool[total_nodes];
    int* locks = new int[total_nodes];
    int* d_r_capacity, *d_parent, *d_flow, *d_locks;
    bool* d_frontier, *d_visited, *d_do_change_capacity;

    size_t locks_size = total_nodes * sizeof(int);

    cudaMalloc((void**)&d_r_capacity, total_nodes * total_nodes * sizeof(int));
    cudaMalloc((void**)&d_parent, total_nodes * sizeof(int));
    cudaMalloc((void**)&d_flow, total_nodes * sizeof(int));
    cudaMalloc((void**)&d_frontier, total_nodes * sizeof(bool));
    cudaMalloc((void**)&d_visited, total_nodes * sizeof(bool));
    cudaMalloc((void**)&d_do_change_capacity, total_nodes * sizeof(bool));
    cudaMalloc((void**)&d_locks, locks_size);

    cudaMemcpy(d_r_capacity, residual, total_nodes * total_nodes * sizeof(int), cudaMemcpyHostToDevice);

    bool found_augmenting_path;
    int max_flow = 0;
    int block_size = 512;
    int grid_size = (total_nodes + block_size - 1) / block_size;

    int counter = 0;
    cudaEventRecord(stopEvent3);
    cudaEventSynchronize(stopEvent3);
    cudaEventElapsedTime(&initmili, startEvent3, stopEvent3);
    totalInitTime += initmili;

    do {
        cudaEventRecord(startEvent3_1);
        for (int i = 0; i < total_nodes; ++i) {
            parent[i] = -1;
            flow[i] = INF;
            locks[i] = 0;
            frontier[i] = (i == source);
            visited[i] = false;
            do_change_capacity[i] = false;
        }

        cudaMemcpy(d_parent, parent, total_nodes * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_flow, flow, total_nodes * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_frontier, frontier, total_nodes * sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(d_visited, visited, total_nodes * sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(d_locks, locks, locks_size, cudaMemcpyHostToDevice);

        cudaEventRecord(stopEvent3_1);
        cudaEventSynchronize(stopEvent3_1);
        cudaEventElapsedTime(&partinitmili, startEvent3_1, stopEvent3_1);
        totalInitTime += partinitmili;

        found_augmenting_path = false;

        while (!found_augmenting_path) {
            cudaEventRecord(startEvent);
            bfsTD <<< grid_size, block_size >>> (d_r_capacity, d_parent, d_flow, d_frontier, d_visited, total_nodes, sink, d_locks);
            cudaDeviceSynchronize();
            bfsBU <<< grid_size, block_size >>> (d_r_capacity, d_parent, d_flow, d_frontier, d_visited, total_nodes, source, d_locks);
            cudaDeviceSynchronize();
            cudaEventRecord(stopEvent);
            cudaEventSynchronize(stopEvent);
            cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
            avgBFSTime += milliseconds;
            bfsCounter++;
            cudaMemcpy(frontier, d_frontier, total_nodes * sizeof(bool), cudaMemcpyDeviceToHost);

            if (sink_reachable(frontier, total_nodes, sink)) {
                found_augmenting_path = true;
                cudaMemcpy(parent, d_parent, total_nodes * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(flow, d_flow, total_nodes * sizeof(int), cudaMemcpyDeviceToHost);
                path_flow = flow[sink];
                max_flow += path_flow;

                int v = sink;
                while (v != source) {
                    int u = parent[v];
                    do_change_capacity[v] = true;
                    v = u;
                }
                cudaMemcpy(d_do_change_capacity, do_change_capacity, total_nodes * sizeof(bool), cudaMemcpyHostToDevice);

                cudaEventRecord(startEvent2);
                cudaAugment_path <<< grid_size, block_size >>> (d_parent, d_do_change_capacity, total_nodes, d_r_capacity, path_flow);
                cudaDeviceSynchronize();
                cudaEventRecord(stopEvent2);
                cudaEventSynchronize(stopEvent2);
                cudaEventElapsedTime(&milliseconds, startEvent2, stopEvent2);
                avgAUGTime += milliseconds;
                augCounter++;
            } else {
                found_augmenting_path = false;
            }
        }
        counter++;
    } while (found_augmenting_path);

    cout << "BFS runs: " << bfsCounter << endl;
    cout << "Average BFS time: " << avgBFSTime / bfsCounter << " ms" << endl;
    cout << "Augmentation runs: " << augCounter << endl;
    cout << "Average augmentation time: " << avgAUGTime / augCounter << " ms" << endl;

    cudaFree(d_r_capacity);
    cudaFree(d_parent);
    cudaFree(d_flow);
    cudaFree(d_frontier);
    cudaFree(d_visited);
    cudaFree(d_do_change_capacity);
    cudaFree(d_locks);
    delete[] residual;
    delete[] parent;
    delete[] flow;
    delete[] frontier;
    delete[] visited;
    delete[] do_change_capacity;
    delete[] locks;

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Average Init time: " << totalInitTime / counter << " ms" << endl;
    cout << "Execution time: " << milliseconds << " ms" << endl;

    return static_cast<float>(max_flow) / 1000.0;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <filename> <total_nodes>" << endl;
        return 1;
    }

    const char* filename = argv[1];
    int total_nodes = stoi(argv[2]);

    float max_flow = edmondskarp(filename, total_nodes);
    cout << "Max Flow: " << max_flow << endl;

    return 0;
}
