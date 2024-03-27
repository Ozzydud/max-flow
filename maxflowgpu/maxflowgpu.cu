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

using namespace std;

#define INF 1e9


void readInput(const char* filename, int total_nodes, int* residual_capacity) {

	ifstream file;
	file.open(filename);

	if (!file) {
        cout <<  "Error reading file!";
        exit(1);
    }

    string line;
    unsigned int source, destination;
    float capacity;

    while (getline(file, line)) {
        if (line.empty()) continue;

        stringstream linestream(line);
        linestream >> source >> destination >> capacity;

        cout << "capacity before " << capacity << " \n";

        int scaledCapacity = static_cast<int>(capacity * 1000);
        residual_capacity[source * total_nodes + destination] = scaledCapacity;

        cout << "capacity after " << residual_capacity[source * total_nodes + destination] << " \n";
    }

    file.close();
}

__global__ void cudaBFS(int *r_capacity, int *parent, int *flow, bool *frontier, bool* visited, int vertices, int sink){
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (!frontier[sink] && Idx < vertices && frontier[Idx]) {
        frontier[Idx] = false;
        visited[Idx] = true;

        for (int i = 0; i < vertices; i++) {
            if (!frontier[i] && !visited[i] && r_capacity[Idx * vertices + i] > 0) {
                frontier[i] = true;
                parent[i] = Idx;
                flow[i] = min(flow[Idx], r_capacity[Idx * vertices + i]);
            }
        }
    }
}


int main() {
    int total_nodes = 19;
    int* residual;

    // Allocating memory for a square matrix representing the graph
    residual = new int[total_nodes * total_nodes];
    memset(residual, 0, sizeof(int) * total_nodes * total_nodes);

    readInput("cage3.mtx", total_nodes, residual);

    int source = 0;
    int sink = total_nodes - 1; // Assuming sink is the last node

    int* parent = new int[total_nodes];
    int* flow = new int[total_nodes];

    for (int i = 0; i < total_nodes; ++i) {
        parent[i] = -1; // Initialize parent array
        flow[i] = INF;  // Initialize flow array with INF
    }

    // Set initial flow from source to 0
    flow[source] = 0;

    int* d_r_capacity, * d_parent, * d_flow;
    bool* frontier, * visited;

    // Allocate memory on device
    cudaMalloc((void**)&d_r_capacity, total_nodes * total_nodes * sizeof(int));
    cudaMalloc((void**)&d_parent, total_nodes * sizeof(int));
    cudaMalloc((void**)&d_flow, total_nodes * sizeof(int));
    cudaMalloc((void**)&frontier, total_nodes * sizeof(bool));
    cudaMalloc((void**)&visited, total_nodes * sizeof(bool));

    // Copy data from host to device
    cudaMemcpy(d_r_capacity, residual, total_nodes * total_nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_parent, parent, total_nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flow, flow, total_nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(frontier, 0, total_nodes * sizeof(bool));
    cudaMemset(visited, 0, total_nodes * sizeof(bool));

    bool sink_reachable = true;
    int max_flow = 0;

    while (sink_reachable) {
        sink_reachable = false;

        // Initialize frontier, visited, parent, and flow arrays
        cudaMemcpy(frontier + source, &d_flow[source], sizeof(bool), cudaMemcpyDeviceToDevice);
        cudaMemcpy(visited + source, &d_flow[source], sizeof(bool), cudaMemcpyDeviceToDevice);
        cudaMemset(d_parent, -1, total_nodes * sizeof(int));

        int block_size = 256;
        int grid_size = (total_nodes + block_size - 1) / block_size;

        // Launch BFS kernel
        cudaBFS<<<grid_size, block_size>>>(d_r_capacity, d_parent, d_flow, frontier, visited, total_nodes, sink);
        cudaDeviceSynchronize();

        // Check if sink is reachable
        cudaMemcpy(&sink_reachable, &frontier[sink], sizeof(bool), cudaMemcpyDeviceToHost);

        if (sink_reachable) {
            int path_flow = INF;

            // Calculate path flow
            for (int v = sink; v != source; v = parent[v]) {
                int u = parent[v];
                path_flow = min(path_flow, residual[u * total_nodes + v]);
            }

            // Update residual capacity and flow along the path
            for (int v = sink; v != source; v = parent[v]) {
                int u = parent[v];
                residual[u * total_nodes + v] -= path_flow;
                residual[v * total_nodes + u] += path_flow;
            }

            max_flow += path_flow;
        }
    }

    cout << "Maximum Flow: " << max_flow << endl;

    // Clean up allocated memory
    delete[] residual;
    delete[] parent;
    delete[] flow;
    cudaFree(d_r_capacity);
    cudaFree(d_parent);
    cudaFree(d_flow);
    cudaFree(frontier);
    cudaFree(visited);

    return 0;
}