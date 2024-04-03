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

    file.close();
}



/* int fordFulkersonCuda(int *row, int *indices, int *data, int source, int sink, int vertices){
    int *d_row, *d_indices, *d_data, *residual, *parent, *flow;
    bool *visited;
    int *queue;
    int *residual;

    // Creating residual graph
    residual = (int*) malloc(vertices); 
    memset(residual, 0, vertices)


    // Allocate all the memory
    cudaMalloc(&d_row, vertices * sizeof(int));
    cudaMalloc(&d_indices, vertices * sizeof(int));
    cudaMalloc(&d_data, vertices * sizeof(int));
    cudaMalloc(&residual, vertices * sizeof(int)); // Same as above - we need to find out how much memory to allocate
    cudaMalloc(&parent, vertices * sizeof(int));
    cudaMalloc(&flow, vertices * sizeof(int));
    cudaMalloc(&visited, vertices * sizeof(bool));
    cudaMalloc(&queue, vertices * sizeof(int));

    // Copy memory to device
    cudaMemcpy(d_row, row, vertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, indices, vertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, data, vertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(residual, d_row, vertices * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize arrays
    cudaMemset(parent, -1, vertices * sizeof(int));
    cudaMemset(flow, 0, vertices * sizeof(int));
    cudaMemset(visited, 0, vertices * sizeof(bool));

    int block_size = 256; //probably not correct
    int num_blocks = (vertices + block_size - 1) / block_size;

    cudaBFS<<<num_blocks, block_size>>>(d_row, d_indices, d_data, source, sink, parent, queue, flow, residual, visited, vertices);

    augmentPath<<<num_blocks, block_size>>>(residual, parent, flow, source, sink, vertices);

    int max_flow;
    cudaMemcpy(&max_flow, &flow[sink], sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_row);
    cudaFree(d_indices);
    cudaFree(d_data);
    cudaFree(residual);
    cudaFree(parent);
    cudaFree(flow);
    cudaFree(visited);
    cudaFree(queue);

    return max_flow;
}


template <typename T>
std::vector<T> readVectorFromFile(const std::string& filePath, float scaleFactor) {
    std::vector<T> values;
    std::ifstream file(filePath);
    float value;
    while (file >> value) {
        // Scale, round, and then convert to integer
        int scaledValue = static_cast<int>(round(value * scaleFactor));
        values.push_back(scaledValue);
    }
    return values;
} */


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

    int* parent = new int[total_nodes];
    int* flow = new int[total_nodes];

    for (int i = 0; i < total_nodes; ++i) {
        parent[i] = -1; // Initialize parent array
        flow[i] = INF;  // Initialize flow array with INF
    }

    // Set initial flow from source to 0
    flow[source] = 0;

    int* d_r_capacity, * d_parent, * d_flow;
    bool* frontier, * visited;

    cout << "test2: " << endl;

    // Allocate memory on device
    cudaMalloc((void**)&d_r_capacity, total_nodes * total_nodes * sizeof(int));
    cudaMalloc((void**)&d_parent, total_nodes * sizeof(int));
    cudaMalloc((void**)&d_flow, total_nodes * sizeof(int));
    cudaMalloc((void**)&frontier, total_nodes * sizeof(bool));
    cudaMalloc((void**)&visited, total_nodes * sizeof(bool));

    cout << "test3: " << d_r_capacity << endl;

    // Copy data from host to device
    cudaMemcpy(d_r_capacity, residual, total_nodes * 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_parent, parent, total_nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flow, flow, total_nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(frontier, 0, total_nodes * sizeof(bool)); // Initialize to false
    cudaMemset(visited, 0, total_nodes * sizeof(bool)); // Initialize to false

    cout << "test4: " << d_r_capacity << endl;

    bool sink_reachable = true;
    int max_flow = 0;

    while (sink_reachable) {
        sink_reachable = false;

        cout << "test5: " << d_r_capacity << endl;

        // Initialize frontier array (only the source node is in the frontier)
        cudaMemset(frontier + source, 0, sizeof(bool));
        cudaMemcpy(frontier + source, &d_flow[source], sizeof(bool), cudaMemcpyDeviceToDevice);
        
        // Initialize visited array (all nodes are not visited)
        cudaMemset(visited, 0, total_nodes * sizeof(bool));

        // Initialize parent array to -1
        cudaMemset(d_parent, -1, total_nodes * sizeof(int));

        int block_size = 256;
        int grid_size = (total_nodes + block_size - 1) / block_size;
        cout << "test6: " << d_r_capacity << endl;
        // Launch BFS kernel
        cudaBFS<<<grid_size, block_size>>>(d_r_capacity, d_parent, d_flow, frontier, visited, total_nodes, sink);
        cudaDeviceSynchronize();
        cout << "test7: " << d_r_capacity << endl;

        // Check if sink is reachable
    cudaMemcpy(&sink_reachable, &frontier[sink], sizeof(bool), cudaMemcpyDeviceToHost);

    cout << "test8: " << d_r_capacity << endl;

    cout << "Maximum Flow: " << max_flow << endl;
    }

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
