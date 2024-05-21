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

__global__ void cudaBFS_TopDown(int *r_capacity, int *parent, int *flow, bool *frontier, bool* visited, int vertices, int source, int* locks, int sink) {
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (!frontier[sink] && Idx < vertices && frontier[Idx]) {
        frontier[Idx] = false;
        visited[Idx] = true;

        for (int i = 0; i < vertices; i++) {
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

__global__ void cudaBFS_BottomUp(int *r_capacity, int *parent, int *flow, bool *frontier, bool* visited, int vertices, int source, int* locks) {
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (!frontier[source] && Idx < vertices && frontier[Idx]) {
        frontier[Idx] = false;
        visited[Idx] = true;
        for (int i = 0; i < vertices; i++) {
            if (!visited[i] && !frontier[i] && r_capacity[i * vertices + Idx] > 0) {
                if (atomicCAS(locks + Idx, 0, 1) == 1 || frontier[Idx]) {
                    continue;
                }
                frontier[Idx] = true;
                locks[Idx] = 0;
                parent[Idx] = i;
                flow[Idx] = min(flow[i], r_capacity[i * vertices + Idx]);
            }
        }
    }
}

__global__ void cudaAugment_pathTD(int* parent, bool* do_change_capacity, int total_nodes, int* r_capacity, int path_flow) {
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(Idx < total_nodes && do_change_capacity[Idx]){
        r_capacity[parent[Idx] * total_nodes + Idx] -= path_flow;
        r_capacity[Idx * total_nodes + parent[Idx]] += path_flow; 
    }    
}
__global__ void cudaAugment_pathBU(int* parent, bool* do_change_capacity, int total_nodes, int* r_capacity, int path_flow) {
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(Idx < total_nodes && do_change_capacity[Idx]){
        r_capacity[Idx * total_nodes + parent[Idx]] -= path_flow; 
        r_capacity[parent[Idx] * total_nodes + Idx] += path_flow;
    }    
}


bool source_reachable(bool* frontier, int total_nodes, int source) {
    
    for (int i = 0; i <= total_nodes-1; ++i) {
        if (frontier[i]) {
            return i == source;  // Source node is reachable from at least one node in the frontier
        }   
    }
    return true;  // Source node is not reachable from any node in the frontier
}

bool sink_reachable(bool* frontier, int total_nodes, int sink){
    for (int i = total_nodes-1; i > -1; --i) {
                if(frontier[i]){
                        return i == sink;
                }
        }
        return true;
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

    int source = 0;
    int sink = total_nodes - 1;
    int path_flow;

    int* parent = new int[total_nodes];
    int* flow = new int[total_nodes];
    bool* frontier = new bool[total_nodes];
    bool* visited = new bool[total_nodes];
    bool* do_change_capacity = new bool[total_nodes];

    flow[source] = 0;
    flow[sink] = 0;
    int* locks = new int[total_nodes];
    int* d_r_capacity, * d_parent, * d_flow, *d_locks;
    bool* d_frontier, * d_visited, *d_do_change_capacity;

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
    int grid_size = ceil(total_nodes * 1.0 / block_size);

    int counter = 0;
    cudaEventRecord(stopEvent3);
    cudaEventSynchronize(stopEvent3);
    cudaEventElapsedTime(&initmili, startEvent3, stopEvent3);
    totalInitTime += initmili;

    bool use_bottom_up = false;
    const float switch_threshold = 0.2;  // 20% threshold
    do {
        cudaEventRecord(startEvent3_1);
        for (int i = 0; i < total_nodes; ++i) {
            parent[i] = -1;
            flow[i] = INF;
            locks[i] = 0;
            frontier[i] = (use_bottom_up && i == sink) || (!use_bottom_up && i == source);
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

        bool did_use_BU = false;
        bool did_not_use_BU = false;
        do {
            cudaEventRecord(startEvent, 0);

            if (use_bottom_up) {
                did_use_BU = true;
                cudaBFS_BottomUp<<<grid_size, block_size>>>(d_r_capacity, d_parent, d_flow, d_frontier, d_visited, total_nodes, source, d_locks);
            } else {
                did_not_use_BU = true;
                cudaBFS_TopDown<<<grid_size, block_size>>>(d_r_capacity, d_parent, d_flow, d_frontier, d_visited, total_nodes, source, d_locks, sink);
            }

            bfsCounter++;
            cudaEventRecord(stopEvent, 0);
            cudaEventSynchronize(stopEvent);

            float miliseconds1 = 0;
            cudaEventElapsedTime(&miliseconds1, startEvent, stopEvent);
            avgBFSTime += miliseconds1;

            cudaMemcpy(frontier, d_frontier, total_nodes * sizeof(bool), cudaMemcpyDeviceToHost);
            cudaMemcpy(visited, d_visited, total_nodes * sizeof(bool), cudaMemcpyDeviceToHost);

            int frontier_size = 0;
            for (int i = 0; i < total_nodes; i++) {
                if (frontier[i]) {
                    frontier_size++;
                }
            }

            // Switch logic based on frontier size
            if (frontier_size > switch_threshold * total_nodes) {
                use_bottom_up = true;
            } else {
                use_bottom_up = false;
            }

        } while ((!use_bottom_up && !sink_reachable(frontier, total_nodes, sink)) || (use_bottom_up && !source_reachable(frontier, total_nodes, source)));

        cudaMemcpy(parent, d_parent, total_nodes * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(flow, d_flow, total_nodes * sizeof(int), cudaMemcpyDeviceToHost);

        found_augmenting_path = false;

        for (int v = 0; v < total_nodes; v++) {
            if ((use_bottom_up && visited[v]) || (!use_bottom_up && frontier[v])) {
                if ((use_bottom_up && v == source) || (!use_bottom_up && v == sink)) {
                    found_augmenting_path = true;
                    path_flow = flow[v];
                    int u = parent[v];
                    while (u != -1) {
                        do_change_capacity[v] = true;
                        v = u;
                        u = parent[v];
                    }
                    break;
                }
            }
        }

        if (found_augmenting_path) {
            augCounter++;
            max_flow += path_flow;
            cudaMemcpy(d_do_change_capacity, do_change_capacity, total_nodes * sizeof(bool), cudaMemcpyHostToDevice);
            cudaEventRecord(startEvent2);
            if(did_not_use_BU){
                cudaAugment_pathTD<<<grid_size, block_size>>>(d_parent, d_do_change_capacity, total_nodes, d_r_capacity, path_flow);
            } else {
                cudaAugment_pathBU<<<grid_size, block_size>>>(d_parent, d_do_change_capacity, total_nodes, d_r_capacity, path_flow);
            }
            cudaEventRecord(stopEvent2);
            cudaEventSynchronize(stopEvent2);

            float milliseconds2 = 0;
            cudaEventElapsedTime(&milliseconds2, startEvent2, stopEvent2);
            avgAUGTime += milliseconds2;
        }

        counter++;
    } while (found_augmenting_path);

    cudaMemcpy(residual, d_r_capacity, total_nodes * total_nodes * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_r_capacity);
    cudaFree(d_parent);
    cudaFree(d_flow);
    cudaFree(d_frontier);
    cudaFree(d_visited);
    cudaFree(d_do_change_capacity);
    cudaFree(d_locks);

    delete[] parent;
    delete[] flow;
    delete[] frontier;
    delete[] visited;
    delete[] do_change_capacity;
    delete[] locks;
    delete[] residual;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaEventDestroy(startEvent2);
    cudaEventDestroy(stopEvent2);

    avgBFSTime /= bfsCounter;
    avgAUGTime /= augCounter;

    cout << "Average BFS Time: " << avgBFSTime << " ms" << endl;
    cout << "Average Augmentation Time: " << avgAUGTime << " ms" << endl;
    cout << "Total Init Time: " << totalInitTime << " ms" << endl;
    cout << max_flow << endl;

    return milliseconds;
}

int main() {
    float ms = 0;
    cout << "cage3.mtx" << endl; 
    
    float test = edmondskarp("cage3.mtx", 5);
    /*
    for (int i = 0; i < 10; i++) {
        ms += edmondskarp("cage3.mtx", 5);
    }

    float ms2 = 0;
    cout << "cage9.mtx" << endl; 
    test = edmondskarp("data/cage9.mtx", 3534);
    for (int i = 0; i < 10; i++) {
        ms2 += edmondskarp("data/cage9.mtx", 3534);
    }

    float ms3 = 0;
    cout << "cage10.mtx" << endl; 
    test = edmondskarp("data/cage10.mtx", 11397);
    for (int i = 0; i < 10; i++) {
        ms3 += edmondskarp("data/cage10.mtx", 11397);
    }

    float ms4 = 0;
    cout << "cage11.mtx" << endl; 
    test = edmondskarp("data/cage11.mtx", 39082);
    for (int i = 0; i < 10; i++) {
        ms4 += edmondskarp("data/cage11.mtx", 39082);
    }

    cout << "cage3.mtx end with an avg speed of " << ms / 10 << endl; 
    cout << "cage9.mtx end with an avg speed of " << ms2 / 10 << endl; 
    cout << "cage10.mtx end with an avg speed of " << ms3 / 10 << endl; 
    cout << "cage11.mtx end with an avg speed of " << ms4 / 10 << endl; 
*/
    return 0;
}
