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


int fordFulkersonCuda(int* r_capacity, int* parent, int* flow, int source, int sink, int vertices){
    int max_flow = 0;

    bool* visited, *frontier;
    int* d_r_capacity, *d_parent, *d_flow
    // allocate memory
    cudaMalloc((void**)&d_r_capacity, vertices * vertices * sizeof(int));
    cudaMalloc((void**)&d_parent, vertices * sizeof(int));
    cudaMalloc((void**)&d_flow, vertices * sizeof(int));
    cudaMalloc((void**)&frontier, vertices * sizeof(bool));
    cudaMalloc((void**)&visited, vertices * sizeof(bool));
    // host -> device
    cudaMemcpy(d_r_capacity, r_capacity, vertices * vertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_parent, parent, vertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flow, flow, vertices * sizeof(int), cudaMemcpyHostToDevice);

    bool sink_reachable = true;

        while (sink_reachable) {
        sink_reachable = false;

        // Initialize frontier, visited, parent, and flow arrays
        cudaMemcpy(frontier, &d_flow[source], vertices * sizeof(bool), cudaMemcpyDeviceToDevice);
        cudaMemcpy(visited, &d_flow[source], vertices * sizeof(bool), cudaMemcpyDeviceToDevice);
        cudaMemset(d_parent, -1, vertices * sizeof(int));

        while (!sink_reachable) {
            // Launch BFS kernel
            int block_size = 256;
            int grid_size = (vertices + block_size - 1) / block_size;
            cudaBFS <<<grid_size, block_size>>>(d_r_capacity, d_parent, d_flow, frontier, visited, vertices, sink);
            cudaDeviceSynchronize();

            // Check if sink is reachable
            cudaMemcpy(&sink_reachable, &frontier[sink], sizeof(bool), cudaMemcpyDeviceToHost);

            if (sink_reachable) {
                int path_flow = INF;

                // Calculate path flow
                for (int v = sink; v != source; v = parent[v]) {
                    int u = parent[v];
                    path_flow = min(path_flow, r_capacity[u * vertices + v]);
                }

                // Update residual capacity and flow along the path
                for (int v = sink; v != source; v = parent[v]) {
                    int u = parent[v];
                    r_capacity[u * vertices + v] -= path_flow;
                    r_capacity[v * vertices + u] += path_flow;
                }

                max_flow += path_flow;
            }
        }
    }
    
    // Copy final flow values back to host
    cudaMemcpy(flow, d_flow, vertices * sizeof(int), cudaMemcpyDeviceToHost);

    // Free allocated memory on device
    cudaFree(d_r_capacity);
    cudaFree(d_parent);
    cudaFree(d_flow);
    cudaFree(frontier);
    cudaFree(visited);

    return max_flow;
}

int main() {
    int total_nodes = 19;
    int* residual;

    // Allocating memory for a square matrix representing the graph
    residual = new int[total_nodes * total_nodes];
    memset(residual, 0, sizeof(int) * total_nodes * total_nodes);

    readInput("cage3.mtx", total_nodes, residual);

    int source = 0;
    int sink = 18; // Assuming sink is node 18

    int* parent = new int[total_nodes];
    int* flow = new int[total_nodes];

    for (int i = 0; i < total_nodes; ++i) {
        parent[i] = -1; // Initialize parent array
        flow[i] = INF;  // Initialize flow array with INF
    }

    // Set initial flow from source to 0
    flow[source] = 0;

    int max_flow = fordFulkersonCuda(residual, parent, flow, source, sink, total_nodes);
    cout << "Maximum Flow: " << max_flow << endl;

    // Clean up allocated memory
    delete[] residual;
    delete[] parent;
    delete[] flow;

    return 0;
}