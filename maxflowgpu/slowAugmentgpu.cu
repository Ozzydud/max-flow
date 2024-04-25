#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

#include <bits/stdc++.h>

// CUDA libraries
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>

using namespace std;

#define INF 1e9

void readInput(const char* filename, int total_nodes, int* residual) {
    ifstream file;
    file.open(filename);

    if (!file) {
        cout << "Error reading file!";
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
            cerr << "Memory allocation failed for residual matrix.";
            exit(EXIT_FAILURE);
        }

        numberOfEdges++;
        residual[source * total_nodes + destination] = scaledCapacity;
    }
    
    cout << "Number of edges in graph is: " << numberOfEdges << endl;
    file.close();
}

__global__ void cudaBFS(int *r_capacity, int *parent, int *flow, bool *frontier, bool* visited, int vertices, int sink, int* locks){
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (!frontier[sink] && Idx < vertices && frontier[Idx]) {
        frontier[Idx] = false;
        visited[Idx] = true;

        for (int i = vertices; i > Idx; i--) {
            if (!frontier[i] && !visited[i] && r_capacity[Idx * vertices + i] > 0) {
                if(atomicCAS(locks+i, 0 , 1) == 1 || frontier[i]){
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
                if(atomicCAS(locks+i, 0 , 1) == 1 || frontier[i]){
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

bool sink_reachable(bool* frontier, int total_nodes, int sink){
    for (int i = total_nodes-1; i > -1; --i) {
        if(frontier[i]){
            return i == sink;
        }
    }
    return true;
}

int main() {
    cudaError_t cudaStatus = cudaSetDevice(4);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?";
        return 1;
    }

    int total_nodes = 39082; // Assuming 3534 or 1107 nodes or 11397 or 39082 or 130228
    int* residual;

    // Allocate memory for residual matrix
    try {
        residual = new int[total_nodes * total_nodes]();
    } catch (const std::bad_alloc& e) {
        cerr << "Failed to allocate memory for the residual matrix: " << e.what() << endl;
        return 1;
    }

    readInput("data/cage11.mtx", total_nodes, residual);

    int source = 0;
    int sink = total_nodes - 1; // Assuming sink is the last node

    int* parent = new int[total_nodes];
    int* flow = new int[total_nodes];
    bool* frontier = new bool[total_nodes];
    bool* visited = new bool [total_nodes];

    // Initialize CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    float avgAUGTime = 0;
    int augCounter = 0;
    cudaEvent_t startEvent2, stopEvent2;
    cudaEventCreate(&startEvent2);
    cudaEventCreate(&stopEvent2);

    // Allocate memory on device
    int* d_r_capacity, * d_parent, * d_flow;
    bool* d_frontier, * d_visited;
    int* d_locks;
    cudaMalloc((void**)&d_parent, total_nodes * sizeof(int));
    cudaMalloc((void**)&d_flow, total_nodes * sizeof(int));
    cudaMalloc((void**)&d_frontier, total_nodes * sizeof(bool));
    cudaMalloc((void**)&d_visited, total_nodes * sizeof(bool));
    cudaMalloc((void**)&d_locks, total_nodes * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_r_capacity, residual, total_nodes * total_nodes * sizeof(int), cudaMemcpyHostToDevice);

    bool found_augmenting_path;
    int max_flow = 0;
    int block_size = 1024;
    int grid_size = ceil(total_nodes * 1.0 / block_size);

    int counter = 0;

    do {
        // Initialize arrays and flags
        for (int i = 0; i < total_nodes; ++i) {
            parent[i] = -1; // Initialize parent array
            flow[i] = INF;  // Initialize flow array with INF
            if (i == source) {
                frontier[i] = true;
            } else {
                frontier[i] = false;
            }
            visited[i] = false;
        }
        
        // Copy arrays to device
        cudaMemcpy(d_parent, parent, total_nodes * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_flow, flow, total_nodes * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_frontier, frontier, total_nodes * sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(d_visited, visited, total_nodes * sizeof(bool), cudaMemcpyHostToDevice);

        // Run BFS kernel until sink is reachable
        while (!sink_reachable(frontier, total_nodes, sink)) {
            cudaBFS<<<grid_size, block_size>>>(d_r_capacity, d_parent, d_flow, d_frontier, d_visited, total_nodes, sink, d_locks);
        }

        // Check if augmenting path found
        found_augmenting_path = frontier[sink];
        if (!found_augmenting_path) {
            break;
        }

        // Copy results from device to host
        cudaMemcpy(parent, d_parent, total_nodes * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(flow, d_flow, total_nodes * sizeof(int), cudaMemcpyDeviceToHost);

        // Calculate path flow and update residual capacities on device
        int path_flow = flow[sink];
        max_flow += path_flow;

        cudaEventRecord(startEvent2, 0);
        for (int i = sink; i != source; i = parent[i]) {
            int residual_capacity_index = parent[i] * total_nodes + i;
            int reverse_residual_capacity_index = i * total_nodes + parent[i];
            residual[residual_capacity_index] -= path_flow;
            residual[reverse_residual_capacity_index] += path_flow;
        }
        
            augCounter++;
        // Stop recording the event
        cudaEventRecord(stopEvent2, 0);
        cudaEventSynchronize(stopEvent2);

        // Calculate elapsed time
        float augmili = 0.0f;
        cudaEventElapsedTime(&augmili, startEvent2, stopEvent2);
        avgAUGTime += augmili;

        counter++;
    } while (found_augmenting_path);

    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Time for BFS and augmenting path: " << milliseconds << " ms\n";
    cout << "Maximum Flow: " << max_flow << endl;

    // Clean up allocated memory
    delete[] residual;
    delete[] parent;
    delete[] flow;
    delete[] frontier;
    delete[] visited;
    cudaFree(d_r_capacity);
    cudaFree(d_parent);
    cudaFree(d_flow);
    cudaFree(d_frontier);
    cudaFree(d_visited);
    cudaFree(d_locks);

    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
