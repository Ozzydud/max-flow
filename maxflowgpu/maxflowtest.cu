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

class Edge {
public:
    int source, destination;
    int capacity;

    Edge(int src, int dest, int cap) : source(src), destination(dest), capacity(cap) {}
};

void readInput(const char* filename, int total_nodes, vector<Edge>& edges) {
    ifstream file;
    file.open(filename);

    if (!file) {
        cout <<  "Error reading file!";
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

        Edge edge(source, destination, scaledCapacity);
        edges.push_back(edge);

        numberOfEdges++;
    }
    
    cout << "Number of edges in graph is: " << numberOfEdges << endl;
    file.close();
}

__global__ void cudaBFS(Edge* edges, int num_edges, int* parent, int* flow, bool* frontier, bool* visited, int vertices, int sink, int* locks) {
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (!frontier[sink] && Idx < vertices && frontier[Idx]) {
        frontier[Idx] = false;
        visited[Idx] = true;

        for (int i = 0; i < num_edges; i++) {
            int source = edges[i].source;
            int destination = edges[i].destination;
            int capacity = edges[i].capacity;

            if (source == Idx && !frontier[destination] && !visited[destination] && capacity > 0) {
                if(atomicCAS(locks + destination, 0 , 1) == 1 || frontier[destination]){
                    continue;
                }

                frontier[destination] = true;
                locks[destination] = 0;
                parent[destination] = Idx;
                flow[destination] = min(flow[Idx], capacity);
            }
        }
    }
}

__global__ void cudaAugment_path(int* parent, bool* do_change_capacity, int total_nodes, Edge* edges, int num_edges, int path_flow) {
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(Idx < total_nodes && do_change_capacity[Idx]){
        for (int i = 0; i < num_edges; i++) {
            int source = edges[i].source;
            int destination = edges[i].destination;
            if (destination == Idx && source == parent[Idx]) {
                edges[i].capacity -= path_flow;
                break;
            }
        }
    }
}

bool sink_reachable(bool* frontier, int total_nodes, int sink) {
    return frontier[sink];
}

int main() {
    cudaError_t cudaStatus = cudaSetDevice(4);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?";
        return 1;
    }

    int total_nodes = 10000; // Assuming 3534 or 1107 nodes or 11397 or 39082 or 130228

    vector<Edge> edges;
    readInput("/home/matthew.jezek/max-flow/main/custom_output_graph1.mtx", total_nodes, edges);
    cout << "data read" << endl;

    int source = 0;
    int sink = total_nodes - 1; // Assuming sink is the last node
    int path_flow;

    int* parent = new int[total_nodes];
    int* flow = new int[total_nodes];
    bool* frontier = new bool[total_nodes];
    bool* visited = new bool[total_nodes];
    bool* do_change_capacity = new bool[total_nodes];
    
    // Set initial flow from source to 0
    flow[source] = 0;
    int* locks = new int[total_nodes];
    Edge* d_edges;
    int* d_parent, * d_flow, * d_locks;
    bool* d_frontier, * d_visited, * d_do_change_capacity;

    size_t edges_size = edges.size() * sizeof(Edge);
    size_t locks_size = total_nodes * sizeof(int);
    
    // Allocate memory on device
    cudaMalloc((void**)&d_edges, edges_size);
    cudaMalloc((void**)&d_parent, total_nodes * sizeof(int));
    cudaMalloc((void**)&d_flow, total_nodes * sizeof(int));
    cudaMalloc((void**)&d_frontier, total_nodes * sizeof(bool));
    cudaMalloc((void**)&d_visited, total_nodes * sizeof(bool));
    cudaMalloc((void**)&d_do_change_capacity, total_nodes * sizeof(bool));
    cudaMalloc((void**)&d_locks, locks_size);

    // Copy data from host to device
    cudaMemcpy(d_edges, edges.data(), edges_size, cudaMemcpyHostToDevice);

    bool found_augmenting_path;
    int max_flow = 0;
    int block_size = 1024;
    int grid_size = ceil(total_nodes * 1.0 / block_size);

    int counter = 0;

    do {
        for (int i = 0; i < total_nodes; ++i) {
            parent[i] = -1; // Initialize parent array
            flow[i] = INF;  // Initialize flow array with INF
            locks[i] = 0;
            if (i == source) {
                frontier[i] = true;
            } else {
                frontier[i] = false;
            }
            visited[i] = false;
            do_change_capacity[i] = false;
        }
   
        cudaMemcpy(d_parent, parent, total_nodes * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_flow, flow, total_nodes * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_frontier, frontier, total_nodes * sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(d_visited, visited, total_nodes * sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(d_locks, locks, locks_size, cudaMemcpyHostToDevice);

        while (!sink_reachable(frontier, total_nodes, sink)) {
            cudaBFS<<<grid_size, block_size>>>(d_edges, edges.size(), d_parent, d_flow, d_frontier, d_visited, total_nodes, sink, d_locks);
            cudaMemcpy(frontier, d_frontier, total_nodes * sizeof(bool), cudaMemcpyDeviceToHost);
        }

        found_augmenting_path = frontier[sink];

        if (!found_augmenting_path) {
            break;
        }

        cudaMemcpy(flow, d_flow, total_nodes * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(parent, d_parent, total_nodes * sizeof(int), cudaMemcpyDeviceToHost);

        path_flow = flow[sink];
        max_flow += path_flow;

        for (int i = sink; i != source; i = parent[i]) {
            do_change_capacity[i] = true;
        }

        cudaMemcpy(d_do_change_capacity, do_change_capacity, total_nodes * sizeof(bool), cudaMemcpyHostToDevice);

        // Launch augmenting path kernel
        cudaAugment_path<<<grid_size, block_size>>>(d_parent, d_do_change_capacity, total_nodes, d_edges, edges.size(), path_flow);

        counter++;
    } while (found_augmenting_path);

    cout << "Counter is: " << counter << endl;

    // Clean up allocated memory
    delete[] parent;
    delete[] flow;
    delete[] locks;
    delete[] frontier;
    delete[] visited;
    delete[] do_change_capacity;
    cudaFree(d_edges);
    cudaFree(d_parent);
    cudaFree(d_flow);
    cudaFree(d_frontier);
    cudaFree(d_visited);
    cudaFree(d_locks);
    cudaFree(d_do_change_capacity);

    return 0;
}