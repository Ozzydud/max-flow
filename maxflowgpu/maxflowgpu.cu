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


void readInput(const char* filename, int total_nodes, int* residual) {

	ifstream file;
	file.open(filename);

	if (!file) {
        cout <<  "Error reading file!";
        exit(1);
    }

    string line;
    int source, destination;
    float capacity;

    while (getline(file, line)) {
        if (line.empty()) continue;

        stringstream linestream(line);
        linestream >> source >> destination >> capacity;

        cout << "Read: Source=" << source << ", Destination=" << destination << ", Capacity=" << capacity << endl;

        source--;
        destination--;

        int scaledCapacity = static_cast<int>(capacity * 1000);
        residual[source * total_nodes + destination] = scaledCapacity;

        cout << "Residual capacity[" << source << "][" << destination << "]: " << residual[source * total_nodes + destination] << endl;
    }
    cout << "hehe" << endl;

    file.close();
}

__global__ void cudaBFS(int *r_capacity, int *parent, int *flow, bool *frontier, bool* visited, int vertices, int sink){
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (!frontier[sink] && Idx < vertices && frontier[Idx]) {
        frontier[Idx] = false;
        visited[Idx] = true;

        for (int i = Idx; i < vertices; i++) {
            if (!frontier[i] && !visited[i] && r_capacity[Idx * vertices + i] > 0) {
                frontier[i] = true;
                parent[i] = Idx;
                flow[i] = min(flow[Idx], r_capacity[Idx * vertices + i]);
            }
        }

        for (int i = 0; i < Idx; i++) {
            if (!frontier[i] && !visited[i] && r_capacity[Idx * vertices + i] > 0) {
                frontier[i] = true;
                parent[i] = Idx;
                flow[i] = min(flow[Idx], r_capacity[Idx * vertices + i]);
            }
        }
    }
}


__global__ void cudaAugment_path(int* parent, bool* do_change_capacity, int total_nodes, int* r_capacity, int path_flow){
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(Idx < total_nodes && do_change_capacity[Idx]){
        r_capacity[parent[Idx] * total_nodes + Idx] -= path_flow;
        r_capacity[Idx * total_nodes + parent[Idx]] += path_flow; 
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
    int total_nodes = 5; // Assuming 5 nodes
    int* residual;

    cout << "test: " << endl;
    // Allocating memory for a square matrix representing the graph
    residual = (int*)malloc(sizeof(int) * total_nodes * total_nodes);
    cout << "test01: " << endl;
    memset(residual, 0, sizeof(int) * total_nodes * total_nodes);
    cout << "test02: " << endl;

    readInput("cage3.mtx", total_nodes, residual);
    cout << residual[2*total_nodes+2] << endl;

        for (int i = 0; i < total_nodes; ++i) {
        for (int j = 0; j < total_nodes; ++j) {
            cout << residual[i * total_nodes + j] << " ";
        }
        cout << endl;
    }

    int source = 0;
    int sink = total_nodes - 1; // Assuming sink is the last node
    int path_flow;

    int* parent = new int[total_nodes];
    int* flow = new int[total_nodes];
    bool* frontier = new bool[total_nodes];
    bool* visited = new bool [total_nodes];
    bool* do_change_capacity = new bool[total_nodes];
    

    // Set initial flow from source to 0
    flow[source] = 0;

    int* d_r_capacity, * d_parent, * d_flow;
    bool* d_frontier, * d_visited, *d_do_change_capacity;

    cout << "test2: " << endl;

    // Allocate memory on device
    cudaMalloc((void**)&d_r_capacity, total_nodes * total_nodes * sizeof(int));
    cudaMalloc((void**)&d_parent, total_nodes * sizeof(int));
    cudaMalloc((void**)&d_flow, total_nodes * sizeof(int));
    cudaMalloc((void**)&d_frontier, total_nodes * sizeof(bool));
    cudaMalloc((void**)&d_visited, total_nodes * sizeof(bool));
    cudaMalloc((void**)&d_do_change_capacity, total_nodes * sizeof(bool));

    cout << "test3: " << d_r_capacity << endl;

    // Copy data from host to device
    cudaMemcpy(d_r_capacity, residual, total_nodes * 3 * sizeof(int), cudaMemcpyHostToDevice);
    cout << "test4: " << d_r_capacity << endl;

    bool found_augmenting_path;
    int max_flow = 0;
    int block_size = 256;
    int grid_size = (total_nodes + block_size - 1) / block_size;

    do{
        cout << "test12: " << d_r_capacity << endl;
        for (int i = 0; i < total_nodes; ++i) {
        parent[i] = -1; // Initialize parent array
        flow[i] = INF;  // Initialize flow array with INF
        if(i == source){
            frontier[i] = true;
        }else{
            frontier[i] = false;
        }

        visited[i] = false;
        do_change_capacity[i] = false;
        }
    cout << "test13: " << d_r_capacity << endl;
        cudaMemcpy(d_parent, parent, total_nodes * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_flow, flow, total_nodes * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_frontier, frontier, total_nodes * sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(d_visited, visited, total_nodes * sizeof(bool), cudaMemcpyHostToDevice);
    cout << "test14: " << d_r_capacity << endl;

        while(!sink_reachable(frontier, total_nodes, sink)){
        cudaBFS<<<grid_size, block_size>>>(d_r_capacity, d_parent, d_flow, d_frontier, d_visited, total_nodes, sink);
        
        cout << "test5: " << d_r_capacity << endl;

        cudaMemcpy(frontier, d_frontier, total_nodes * sizeof(bool), cudaMemcpyDeviceToHost);
        }

        found_augmenting_path = frontier[sink];

        if(!found_augmenting_path){
            break;
        }

        cudaMemcpy(flow, d_flow, total_nodes * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(parent, d_parent, total_nodes * sizeof(int), cudaMemcpyDeviceToHost);

        path_flow = flow[sink];
        max_flow += path_flow;

        for(int i = sink; i != source; i = parent[i]){
			do_change_capacity[i] = true;
            i--;
		}

        cudaMemcpy(d_do_change_capacity, do_change_capacity, total_nodes * sizeof(bool), cudaMemcpyHostToDevice);



        
        cout << "test6: " << d_r_capacity << endl;
        // Launch BFS kernel
        cudaAugment_path<<< grid_size, block_size >>>(d_parent, d_do_change_capacity, total_nodes, d_r_capacity, path_flow);

        cout << "Maximum Flow: " << max_flow << endl;

        cout << "test7: " << d_r_capacity << endl;
        


    }while(found_augmenting_path);
    
    cout << "test8: " << d_r_capacity << endl;

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