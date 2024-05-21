#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#define N 5 // Number of nodes (example size)

using namespace std;

// CUDA error checking
#define cudaCheckError() { \
    cudaError_t e = cudaGetLastError(); \
    if (e != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(e)); \
        exit(1); \
    } \
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

__global__ void topDownBFS(int *adjMatrix, bool *frontier, bool *newFrontier, int *visited, int n, int *parent, int *flow) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < n && frontier[u]) {
        for (int v = 0; v < n; ++v) {
            if (adjMatrix[u * n + v] > 0 && !visited[v]) {
                newFrontier[v] = true;
                visited[v] = true;
                parent[v] = u;
                flow[v] = min(flow[u], adjMatrix[u * n + v]); // Calculate flow along the path
            }
        }
    }
}

__global__ void bottomUpBFS(int *adjMatrix, bool *frontier, bool *newFrontier, int *visited, int n, int *parent, int *flow) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n && !visited[v]) {
        for (int u = 0; u < n; ++u) {
            if (adjMatrix[u * n + v] > 0 && frontier[u]) {
                newFrontier[v] = true;
                visited[v] = true;
                parent[v] = u;
                flow[v] = min(flow[u], adjMatrix[u * n + v]); // Calculate flow along the path
                break;
            }
        }
    }
}

void bfs(int *adjMatrix, int n, int source, int sink, int &maxFlow) {
    bool *frontier, *newFrontier;
    int *visited, *parent, *flow;

    cudaMallocManaged(&frontier, n * sizeof(bool));
    cudaMallocManaged(&newFrontier, n * sizeof(bool));
    cudaMallocManaged(&visited, n * sizeof(int));
    cudaMallocManaged(&parent, n * sizeof(int));
    cudaMallocManaged(&flow, n * sizeof(int));

    for (int i = 0; i < n; ++i) {
        frontier[i] = false;
        newFrontier[i] = false;
        visited[i] = 0;
        parent[i] = -1;
        flow[i] = 0;
    }
    frontier[source] = true;
    visited[source] = true;
    flow[source] = INT_MAX; // Set initial flow to maximum value

    bool isTopDown = true;
    int frontierSize = 1;

    while (frontierSize > 0 && !visited[sink]) {
        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;

        if (isTopDown) {
            topDownBFS<<<numBlocks, blockSize>>>(adjMatrix, frontier, newFrontier, visited, n, parent, flow);
        } else {
            bottomUpBFS<<<numBlocks, blockSize>>>(adjMatrix, frontier, newFrontier, visited, n, parent, flow);
        }

        cudaDeviceSynchronize();
        cudaCheckError();

        // Count new frontier size and decide if we should switch approach
        frontierSize = 0;
        for (int i = 0; i < n; ++i) {
            frontier[i] = newFrontier[i];
            newFrontier[i] = false;
            if (frontier[i]) {
                frontierSize++;
            }
        }

        if (frontierSize > n / 10) { // Example threshold for switching
            isTopDown = !isTopDown;
        }
    }

    // Calculate flow along augmenting path and update residual capacities
    int pathFlow = flow[sink];
    if (pathFlow == 0) return;

    maxFlow += pathFlow;

    int v = sink;
    while (parent[v] != -1) {
        int u = parent[v];
        adjMatrix[u * n + v] -= pathFlow;
        adjMatrix[v * n + u] += pathFlow; // Update backward edge
        v = u;
    }

    cudaFree(frontier);
    cudaFree(newFrontier);
    cudaFree(visited);
    cudaFree(parent);
    cudaFree(flow);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <input_file>" << endl;
        return 1;
    }

    const char* filename = argv[1];

    int *adjMatrix;
    cudaMallocManaged(&adjMatrix, N * N * sizeof(int));
    memset(adjMatrix, 0, N * N * sizeof(int)); // Initialize the adjacency matrix with zeros

    readInput(filename, N, adjMatrix);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    int maxFlow = 0;
    while (true) {
        bool *frontier, *newFrontier;
        int *visited, *parent, *flow;
        bool foundPath = false;

        cudaMallocManaged(&frontier, N * sizeof(bool));
        cudaMallocManaged(&newFrontier, N * sizeof(bool));
        cudaMallocManaged(&visited, N * sizeof(int));
        cudaMallocManaged(&parent, N * sizeof(int));
        cudaMallocManaged(&flow, N * sizeof(int));

        for (int i = 0; i < N; ++i) {
            frontier[i] = false;
            newFrontier[i] = false;
            visited[i] = 0;
            parent[i] = -1;
            flow[i] = 0;
        }
        frontier[0] = true; // Assume source is node 0
        visited[0] = true;
        flow[0] = INT_MAX; // Set initial flow to maximum value

        bool isTopDown = true;
        int frontierSize = 1;

        while (frontierSize > 0 && !visited[N - 1]) { // Assume sink is node N-1
            int blockSize = 256;
            int numBlocks = (N + blockSize - 1) / blockSize;

            if (isTopDown) {
                topDownBFS<<<numBlocks, blockSize>>>(adjMatrix, frontier, newFrontier, visited, N, parent, flow);
                cout << "TD" << endl;
            } else {
                cout << "BU" << endl;
                bottomUpBFS<<<numBlocks, blockSize>>>(adjMatrix, frontier, newFrontier, visited, N, parent, flow);
            }

            cudaDeviceSynchronize();
            cudaCheckError();

            // Count new frontier size and decide if we should switch approach
            frontierSize = 0;
            for (int i = 0; i < N; ++i) {
                frontier[i] = newFrontier[i];
                newFrontier[i] = false;
                if (frontier[i]) {
                    frontierSize++;
                }
            }

            if (frontierSize > N / 10) { // Example threshold for switching
                isTopDown = !isTopDown;
            }
        }

        if (!visited[N - 1]) break; // No augmenting path found

        int pathFlow = flow[N - 1];
        maxFlow += pathFlow;

        int v = N - 1;
        while (parent[v] != -1) {
            int u = parent[v];
            adjMatrix[u * N + v] -= pathFlow;
            adjMatrix[v * N + u] += pathFlow; // Update backward edge
            v = u;
        }

        cudaFree(frontier);
        cudaFree(newFrontier);
        cudaFree(visited);
        cudaFree(parent);
        cudaFree(flow);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "Maximum Flow: " << maxFlow << endl;
    cout << "Time elapsed: " << milliseconds << " ms" << endl;

    cudaFree(adjMatrix);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
