#include <iostream>
#include <vector>
#include <queue>
#include <cstring>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define INF INT_MAX

using namespace std;

// CUDA kernel for the top-down BFS
__global__ void bfsTopDown(int *rGraph, int *parent, bool *visited, int *frontier, int vertices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < vertices && frontier[idx]) {
        frontier[idx] = false;
        for (int v = 0; v < vertices; ++v) {
            if (!visited[v] && rGraph[idx * vertices + v] > 0) {
                visited[v] = true;
                parent[v] = idx;
                frontier[v] = true;
            }
        }
    }
}

// CUDA kernel for the bottom-up BFS
__global__ void bfsBottomUp(int *rGraph, int *parent, bool *visited, int *frontier, int vertices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < vertices && !visited[idx]) {
        for (int u = 0; u < vertices; ++u) {
            if (frontier[u] && rGraph[u * vertices + idx] > 0) {
                visited[idx] = true;
                parent[idx] = u;
                frontier[idx] = true;
                break;
            }
        }
    }
}

__host__ bool bfs(int *rGraph, int s, int t, int *parent, int vertices) {
    bool *visited;
    bool *d_visited;
    int *d_parent;
    int *frontier;
    int *d_frontier;
    int *d_rGraph;

    visited = new bool[vertices];
    memset(visited, 0, sizeof(bool) * vertices);

    cudaMalloc((void**)&d_visited, sizeof(bool) * vertices);
    cudaMalloc((void**)&d_parent, sizeof(int) * vertices);
    cudaMalloc((void**)&d_frontier, sizeof(int) * vertices);
    cudaMalloc((void**)&d_rGraph, sizeof(int) * vertices * vertices);

    cudaMemcpy(d_rGraph, rGraph, sizeof(int) * vertices * vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(d_visited, visited, sizeof(bool) * vertices, cudaMemcpyHostToDevice);

    frontier = new int[vertices];
    memset(frontier, 0, sizeof(int) * vertices);
    frontier[s] = 1;

    cudaMemcpy(d_frontier, frontier, sizeof(int) * vertices, cudaMemcpyHostToDevice);

    int blockSize = 512;
    int numBlocks = (vertices + blockSize - 1) / blockSize;

    while (true) {
        cudaMemcpy(d_frontier, frontier, sizeof(int) * vertices, cudaMemcpyHostToDevice);

        bool *h_visited = new bool[vertices];
        cudaMemcpy(h_visited, d_visited, sizeof(bool) * vertices, cudaMemcpyDeviceToHost);

        bool topDown = true;
        for (int i = 0; i < vertices; i++) {
            if (frontier[i] && !h_visited[i]) {
                topDown = false;
                break;
            }
        }
        delete[] h_visited;

        if (topDown) {
            bfsTopDown<<<numBlocks, blockSize>>>(d_rGraph, d_parent, d_visited, d_frontier, vertices);
        } else {
            bfsBottomUp<<<numBlocks, blockSize>>>(d_rGraph, d_parent, d_visited, d_frontier, vertices);
        }

        cudaMemcpy(visited, d_visited, sizeof(bool) * vertices, cudaMemcpyDeviceToHost);

        bool done = visited[t];
        if (done) break;
    }

    cudaMemcpy(parent, d_parent, sizeof(int) * vertices, cudaMemcpyDeviceToHost);

    delete[] visited;
    delete[] frontier;
    cudaFree(d_visited);
    cudaFree(d_parent);
    cudaFree(d_frontier);
    cudaFree(d_rGraph);

    return visited[t];
}

int edmondsKarp(int *graph, int s, int t, int vertices) {
    int u, v;
    int *rGraph = new int[vertices * vertices];

    for (u = 0; u < vertices; u++)
        for (v = 0; v < vertices; v++)
            rGraph[u * vertices + v] = graph[u * vertices + v];

    int *parent = new int[vertices];
    int max_flow = 0;

    while (bfs(rGraph, s, t, parent, vertices)) {
        int path_flow = INF;
        for (v = t; v != s; v = parent[v]) {
            u = parent[v];
            path_flow = min(path_flow, rGraph[u * vertices + v]);
        }

        for (v = t; v != s; v = parent[v]) {
            u = parent[v];
            rGraph[u * vertices + v] -= path_flow;
            rGraph[v * vertices + u] += path_flow;
        }

        max_flow += path_flow;
    }

    delete[] rGraph;
    delete[] parent;

    return max_flow;
}

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

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <input_file>" << endl;
        return 1;
    }

    const char* filename = argv[1];
    int vertices = 6; // You can change this based on your input graph
    int *graph = new int[vertices * vertices];
    memset(graph, 0, sizeof(int) * vertices * vertices);
    
    readInput(filename, vertices, graph);

    cout << "The maximum possible flow is " << edmondsKarp(graph, 0, vertices-1, vertices) << endl;

    delete[] graph;
    return 0;
}
