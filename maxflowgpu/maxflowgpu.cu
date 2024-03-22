#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// CUDA libraries
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define INF 1e9


void readInput(const char* filename, int total_nodes, int* residual_capacity) {

	ifstream file;
	file.open(filename);

	if (!file) {
        cout <<  "Error reading file!";
        exit(1);
    }

    string line;
    u_int source, destination;
    u_short capacity;

    while (file) {

        getline(file, line);

        if (line.empty()) {
            continue;
        }

        std::stringstream linestream(line);
        linestream >> source >> destination >> capacity*1000f;
        residual_capacity[source * total_nodes + destination] = capacity;
        printf("capacity after %d \n", capacity);
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
    int *residual;

    // Creating residual graph
    residual = (int*) malloc(19); 
    memset(residual, 0, 19);

    readInput('cage3.mtx', 19, residual);
    return 0;
}
