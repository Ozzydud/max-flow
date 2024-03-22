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

    if(!frontier[sink] && Idx < vertices && frontier[Idx]){

        frontier[Idx] = false;
        visited[Idx] = true;

        int *neighbour_parent;
        int *neighbour_flow;
        int capacity;

        for (int i = Idx; i<vertices; i++){
            capacity = r_capacity[Idx * vertices + i];

            if(frontiter[i] || visited[i] || capacity <= 0){
                continue; // If we have already seen the neighbours and perfomed the iteration on this, we move on
            }

            frontier[i] = true;

            // maybe otherwise
            neighbour_parent + i = Idx; 
            neighbour_flow + i = min(flow[idx], capacity)

        }


        
    }

}

int main() {
/*     float scaleFactor = 1000.0f;

    std::vector<int> data = readVectorFromFile<int>("output_csr_data.txt", scaleFactor);
    std::vector<int> colIndices = readVectorFromFile<int>("output_csr_col_indices.txt", 1);
    std::vector<int> csrRowPtr = readVectorFromFile<int>("output_csr_row_ptr.txt", 1);


    int V = csrRowPtr.size() - 1; // Number of vertices
    int s = 0; // Source
    int t = 5; // Sink

    // Convert vectors to pointers
    int *d_csrRowPtr = &csrRowPtr[0];
    int *d_colIndices = &colIndices[0];
    int *d_data = &data[0];

    int max_flow = fordFulkersonCuda(d_csrRowPtr, d_colIndices, d_data, s, t, V);
    std::cout << "The maximum possible flow is " << max_flow << std::endl; */



    int total_nodes = 19;
    int *residual;

    // Allocating memory for a square matrix representing the graph
    residual = new int[total_nodes * total_nodes];
    memset(residual, 0, sizeof(int) * total_nodes * total_nodes);

    readInput("cage3.mtx", total_nodes, residual);

    // Remember to free the allocated memory
    delete[] residual;
    

    return 0;
}
