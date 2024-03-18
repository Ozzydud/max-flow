#include <iostream>
#include <fstream>
#include <vector>

// CUDA libraries
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define V 5
#define BLOCK_SIZE 5

// CUDA kernel for BFS traversal
__global__ void cuda_bfs(int* rGraph, bool* visited, int* parent, int t, bool* found) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (visited[idx] || *found)
        return;

    visited[idx] = true;
    parent[idx] = -1;

    if (idx == t) {
        *found = true;
        return;
    }

    for (int v = 0; v < V; v++) {
        if (!visited[v] && rGraph[idx * V + v] > 0) {
            parent[v] = idx;
        }
    }
}

// CUDA kernel for calculating path flow
__global__ void cuda_calculate_path_flow(int* rGraph, int* parent, int* path_flow, int s, int t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == t) {
        int v = idx;
        int u = parent[v];
        *path_flow = rGraph[u * V + v];
        while (u != s) {
            v = u;
            u = parent[v];
            *path_flow = min(*path_flow, rGraph[u * V + v]);
        }
    }
}

// CUDA kernel for updating residual capacities
__global__ void cuda_update_residual_capacities(int* rGraph, int* parent, int* path_flow, int s, int t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == t) {
        int v = idx;
        int u = parent[v];
        while (u != s) {
            v = u;
            u = parent[v];
            rGraph[u * V + v] -= *path_flow;
            rGraph[v * V + u] += *path_flow;
        }
    }
}

// Returns the maximum flow from s to t in the given graph
int fordFulkerson(int* graph, int s, int t) {
    int* rGraph;
    cudaMalloc(&rGraph, V * V * sizeof(int));
    cudaMemcpy(rGraph, graph, V * V * sizeof(int), cudaMemcpyHostToDevice);

    bool* visited;
    cudaMalloc(&visited, V * sizeof(bool));
    bool* d_found;
    cudaMalloc(&d_found, sizeof(bool));
    int* parent;
    cudaMalloc(&parent, V * sizeof(int));

    int max_flow = 0;
    bool found_path;

    do {
        found_path = false;
        cudaMemset(visited, 0, V * sizeof(bool));
        cudaMemset(d_found, false, sizeof(bool));

        cuda_bfs<<<(V + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(rGraph, visited, parent, t, d_found);
        cudaDeviceSynchronize();

        bool h_found;
        cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);

        if (h_found) {
            int* path_flow;
            cudaMalloc(&path_flow, sizeof(int));

            cuda_calculate_path_flow<<<1, 1>>>(rGraph, parent, path_flow, s, t);
            cudaDeviceSynchronize();

            int h_path_flow;
            cudaMemcpy(&h_path_flow, path_flow, sizeof(int), cudaMemcpyDeviceToHost);

            cuda_update_residual_capacities<<<1, 1>>>(rGraph, parent, path_flow, s, t);
            cudaDeviceSynchronize();

            max_flow += h_path_flow;
            cudaFree(path_flow);
        }
    } while (found_path);

    cudaFree(rGraph);
    cudaFree(visited);
    cudaFree(d_found);
    cudaFree(parent);

    return max_flow;
}

int main() {
    std::vector<std::vector<int>> adjMatrix = {
        {667, 367, 300, 367, 300},
        {100, 533, 0, 200, 0},
        {122, 0, 578, 0, 244},
        {50, 100, 0, 283, 183},
        {61, 0, 122, 150, 272}
    };

    int* graph = new int[V * V];
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            graph[i * V + j] = adjMatrix[i][j];
        }
    }

    int max_flow = fordFulkerson(graph, 0, 4);

    std::cout << "The maximum possible flow is " << max_flow << std::endl;

    delete[] graph;

    return 0;
}
