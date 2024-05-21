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

__global__ void cudaBFS_TopDown(int *r_capacity, int *parent, int *flow, bool *frontier, bool* visited, int vertices, int source, int* locks) {
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (Idx < vertices && frontier[Idx]) {
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
    if (Idx < vertices && !visited[Idx]) {
        for (int i = 0; i < vertices; i++) {
            if (frontier[i] && r_capacity[i * vertices + Idx] > 0) {
                if (atomicCAS(locks + Idx, 0, 1) == 1 || frontier[Idx]) {
                    continue;
                }
                frontier[Idx] = true;
                locks[Idx] = 0;
                parent[Idx] = i;
                flow[Idx] = min(flow[i], r_capacity[i * vertices + Idx]);
                visited[Idx] = true;
            }
        }
    }
}

__global__ void cudaAugment_path(int* parent, bool* do_change_capacity, int total_nodes, int* r_capacity, int path_flow) {
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (Idx < total_nodes && do_change_capacity[Idx]) {
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

        int old_work = 0;
        int new_work = 0;
        cout << "test5" << endl;
        cout << use_bottom_up << !sink_reachable(frontier, total_nodes, sink) << !use_bottom_up << !source_reachable(frontier, total_nodes, source) << endl;
        while ((use_bottom_up && !sink_reachable(frontier, total_nodes, sink)) || (!use_bottom_up && !source_reachable(frontier, total_nodes, source))) {
            cudaEventRecord(startEvent, 0);
            cout << "test0" << endl;
            if (use_bottom_up) {
                cout << "test1" << endl;
                cudaBFS_BottomUp<<<grid_size, block_size>>>(d_r_capacity, d_parent, d_flow, d_frontier, d_visited, total_nodes, source, d_locks);
            } else {
                cout << "test2" << endl;
                cudaBFS_TopDown<<<grid_size, block_size>>>(d_r_capacity, d_parent, d_flow, d_frontier, d_visited, total_nodes, source, d_locks);
            }
            cout << "test3" << endl;
            bfsCounter++;
            cudaEventRecord(stopEvent, 0);
            cudaEventSynchronize(stopEvent);

            float miliseconds1 = 0;
            cudaEventElapsedTime(&miliseconds1, startEvent, stopEvent);
            avgBFSTime += miliseconds1;

            cudaMemcpy(frontier, d_frontier, total_nodes * sizeof(bool), cudaMemcpyDeviceToHost);
            cudaMemcpy(visited, d_visited, total_nodes * sizeof(bool), cudaMemcpyDeviceToHost);

            new_work = 0;
            for (int i = 0; i < total_nodes; i++) {
                if (visited[i]) {
                    new_work++;
                }
            }

            if (new_work > 2 * old_work) {
                use_bottom_up = !use_bottom_up;
            }
            old_work = new_work;
        }

        found_augmenting_path = frontier[source];

        if (!found_augmenting_path) {
            break;
        }

        cudaMemcpy(flow, d_flow, total_nodes * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(parent, d_parent, total_nodes * sizeof(int), cudaMemcpyDeviceToHost);

        path_flow = flow[source];
        max_flow += path_flow;
        cout << max_flow << endl;

        for (int i = source; i != sink; i = parent[i]) {
            do_change_capacity[i] = true;
        }

        cudaMemcpy(d_do_change_capacity, do_change_capacity, total_nodes * sizeof(bool), cudaMemcpyHostToDevice);

        cudaEventRecord(startEvent2, 0);
        cudaAugment_path<<<grid_size, block_size>>>(d_parent, d_do_change_capacity, total_nodes, d_r_capacity, path_flow);
        augCounter++;
        cudaEventRecord(stopEvent2, 0);
        cudaEventSynchronize(stopEvent2);

        float augmili = 0.0f;
        cudaEventElapsedTime(&augmili, startEvent2, stopEvent2);
        avgAUGTime += augmili;

        counter++;
    } while (counter != 3);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "Time for BFS and augmenting path: " << milliseconds << " ms\n";
    cout << "Average BFS time is: " << avgBFSTime / bfsCounter << "ms\n";
    cout << "Total time BFS is: " << avgBFSTime << "ms\n";
    cout << "Total AUG time is " << avgAUGTime << "ms\n";
    cout << "Average AUG time is: " << avgAUGTime / augCounter << "ms\n";
    cout << "Total init time is: " << totalInitTime << "ms\n";
    cout << "Maximum Flow: " << max_flow << endl;

    delete[] residual;
    delete[] parent;
    delete[] flow;
    delete[] locks;
    delete[] frontier;
    delete[] visited;
    delete[] do_change_capacity;
    cudaFree(d_r_capacity);
    cudaFree(d_parent);
    cudaFree(d_flow);
    cudaFree(d_frontier);
    cudaFree(d_visited);
    cudaFree(d_locks);
    cudaFree(d_do_change_capacity);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(stopEvent);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent2);
    cudaEventDestroy(startEvent2);
    cudaEventDestroy(stopEvent3);
    cudaEventDestroy(startEvent3);
    cudaEventDestroy(startEvent3_1);
    cudaEventDestroy(stopEvent3_1);

    return milliseconds;
}

int main() {
    float ms = 0;
    cout << "cage3.mtx" << endl; 
    float test = edmondskarp("cage3.mtx", 5);
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

    return 0;
}
