#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <climits>


using namespace std;

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

__global__ void topDownBFS(int *adjMatrix, bool *frontier, bool *newFrontier, int *visited, int n, int *parent, int *flow, int* locks) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < n && frontier[u]) {
        for (int v = 0; v < n; ++v) {
            if (adjMatrix[u * n + v] > 0 && !visited[v]) {
                if (!visited[v]) {
                    newFrontier[v] = true;
                    visited[v] = true;
                    parent[v] = u;
                    flow[v] = min(flow[u], adjMatrix[u * n + v]); // Calculate flow along the path
                }
            }
        }
    }
}

__global__ void bottomUpBFS(int *adjMatrix, bool *frontier, bool *newFrontier, int *visited, int n, int *parent, int *flow, int* locks) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n && !visited[v]) {
        for (int u = 0; u < n; ++u) {
            if (adjMatrix[u * n + v] > 0 && frontier[u]) {
                if (!visited[v]) {
                    newFrontier[v] = true;
                    visited[v] = true;
                    parent[v] = u;
                    flow[v] = min(flow[u], adjMatrix[u * n + v]); // Calculate flow along the path
                }
                break;
            }
        }
    }
}

float edmondskarp(char* filename, int total_nodes){
    int *residual;
    try {
	residual = new int[total_nodes * total_nodes]();
    } catch (const std::bad_alloc& e) {
	    std::cerr << "Failed to allocate memory for the residual matrix: " << e.what() << std::endl;
	    return 1;
    }

    readInput(filename, total_nodes, residual);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    int maxFlow = 0;

    // Allocate memory for BFS-related arrays once
    bool *frontier, *newFrontier;
    int *visited, *parent, *flow, *locks;
    int* adjMatrix;
    if (cudaMallocManaged(&locks, total_nodes * sizeof(int)) != cudaSuccess ||
        cudaMallocManaged(&adjMatrix, total_nodes * total_nodes * sizeof(int)) != cudaSuccess ||
        cudaMallocManaged(&frontier, total_nodes * sizeof(bool)) != cudaSuccess ||
        cudaMallocManaged(&newFrontier, total_nodes * sizeof(bool)) != cudaSuccess ||
        cudaMallocManaged(&visited, total_nodes * sizeof(int)) != cudaSuccess ||
        cudaMallocManaged(&parent, total_nodes * sizeof(int)) != cudaSuccess ||
        cudaMallocManaged(&flow, total_nodes * sizeof(int)) != cudaSuccess) {
        cerr << "Memory allocation failed" << endl;
        return 1;
    }
    cudaMemcpy(adjMatrix, residual, total_nodes * total_nodes * sizeof(int), cudaMemcpyHostToDevice);


    while (true) {
        // Reset state for the new BFS iteration
        for (int i = 0; i < total_nodes; ++i) {
            frontier[i] = false;
            newFrontier[i] = false;
            visited[i] = 0;
            parent[i] = -1;
            flow[i] = 0;
            locks[i] = 0;
        }
        frontier[0] = true; // Assume source is node 0
        visited[0] = true;
        flow[0] = INT_MAX; // Set initial flow to maximum value

        bool isTopDown = true;
        int frontierSize = 1;

        while (frontierSize > 0 && !visited[total_nodes - 1]) { // Assume sink is node N-1
            int blockSize = 512;
            int numBlocks = (total_nodes + blockSize - 1) / blockSize;

            if (isTopDown) {
                topDownBFS<<<numBlocks, blockSize>>>(adjMatrix, frontier, newFrontier, visited, total_nodes, parent, flow, locks);
            } else {
                bottomUpBFS<<<numBlocks, blockSize>>>(adjMatrix, frontier, newFrontier, visited, total_nodes, parent, flow, locks);
            }

            cudaDeviceSynchronize();
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                cerr << "CUDA error: " << cudaGetErrorString(err) << endl;
                return 1;
            }

            // Count new frontier size and decide if we should switch approach
            frontierSize = 0;
            for (int i = 0; i < total_nodes; ++i) {
                frontier[i] = newFrontier[i];
                newFrontier[i] = false;
                if (frontier[i]) {
                    frontierSize++;
                }
            }

            if (frontierSize > total_nodes / 10) { // Example threshold for switching
                isTopDown = !isTopDown;
            }
        }

        if (!visited[total_nodes - 1]) break; // No augmenting path found

        int pathFlow = flow[total_nodes - 1];
        maxFlow += pathFlow;

        int v = total_nodes - 1;
        while (parent[v] != -1) {
            int u = parent[v];
            adjMatrix[u * total_nodes + v] -= pathFlow;
            adjMatrix[v * total_nodes + u] += pathFlow; // Update backward edge
            v = u;
        }
    }

    // Free allocated memory
    cudaFree(frontier);
    cudaFree(newFrontier);
    cudaFree(visited);
    cudaFree(parent);
    cudaFree(flow);
    cudaFree(locks);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "Maximum Flow: " << maxFlow << endl;
    cout << "Time elapsed: " << milliseconds << " ms" << endl;

    cudaFree(adjMatrix);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return milliseconds;
}

int main() {
    float ms = 0;
    cout << "cage3.mtx" << endl; 
    float test = edmondskarp("cage3.mtx", 5);
    for(int i = 0; i<10; i++){
        ms += edmondskarp("cage3.mtx", 5);
    }

    


    float ms2 = 0;
    cout << "cage9.mtx" << endl; 
    test = edmondskarp("data/cage9.mtx", 3534);
    for(int i = 0; i<10; i++){
        ms2 += edmondskarp("data/cage9.mtx", 3534);
    }

    

    float ms3 = 0;
    cout << "cage10.mtx" << endl; 
    test = edmondskarp("data/cage10.mtx", 11397);
    for(int i = 0; i<10; i++){
        ms3 += edmondskarp("data/cage10.mtx", 11397);
    }

    

    float ms4 = 0;
    cout << "cage11.mtx" << endl; 
    test = edmondskarp("data/cage11.mtx", 39082);
    for(int i = 0; i<10; i++){
        ms4 += edmondskarp("data/cage11.mtx", 39082);
    }

    cout << "cage3.mtx end with a avg speed of" << ms/10 << endl; 
    cout << "cage9.mtx end with a avg speed of" << ms2/10 << endl; 
    cout << "cage10.mtx end with a avg speed of" << ms3/10 << endl; 
    cout << "cage11.mtx end with a avg speed of" << ms4/10 << endl; 
    
    

}
