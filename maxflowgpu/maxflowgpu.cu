#include <iostream>
#include <fstream>
#include <vector>

// CUDA libraries
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define INF 1e9

// BFS
__global__ void cudaBFS (int *row, int *indices, int *data,
                         int source, int sink, int *parent, int *queue, int *flow, int *residual, bool *visited, int vertices){
    int tid = blockIdx.x * blockDim.x * threadIdx.x; //Finding thread ID
    if(visited[tid] == false && vertices > tid){ //Mark as visited and add tid to the queue
        queue[tid] = tid;
        visited[tid] = true;
        parent[tid] = -1;
    }

     __syncthreads(); // Not optimal - we need to wait for all threads before we do BFS

     while (!visited[sink] && !visited[source]) { //We keep going as long as we have not visited both sink and source
            // Needs changing to fit with our data ---- ALL OF THE BELOW
            for (int i = row[tid]; i < row[tid + 1]; ++i) {
            int v = indices[i]; // Get the destination vertex
            if (!visited[v] && residual[i] > 0) {
                // Process neighboring vertices
                    queue[v] = tid;
                    visited[v] = true;
                    parent[v] = tid;
            }
        }
         __syncthreads();
     }

}

//AUGMENTED PATHS
__global__ void augmentPath(int *residual, int *parent, int *flow, 
                            int source, int sink, int vertices){
    int tid = blockIdx.x * blockDim.x * threadIdx.x; //Finding thread ID
    if(tid<vertices && parent[tid] != -1){ //if == -1, it was not reached in BFS
        int min_flow = INF;
        int current = tid;
        while (current != source) {
            int current_parent = parent[current];
            // Needs changing to follow data structure
            min_flow = min(min_flow, residual[current_parent * vertices + current]);
            current = current_parent;
        }

        current = tid;
        while(current != source){
            int current_parent = parent[current];
            residual[current_parent * vertices + current] -= min_flow;
            residual[current * vertices + current_parent] += min_flow;
            current = current_parent;
        }
        flow[tid] += min_flow;
    }
}

int fordFulkersonCuda(int *row, int *indices, int *data, int source, int sink, int vertices){
    int *d_row, *d_indices, *d_data, *residual, *parent, *flow;
    bool *visited;
    int *queue;

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
}


int main() {
    float scaleFactor = 1000.0f;

    std::vector<int> data = readVectorFromFile<int>("output_csr_data.txt", scaleFactor);
    std::vector<int> colIndices = readVectorFromFile<int>("output_csr_col_indices.txt", 1);
    std::vector<int> csrRowPtr = readVectorFromFile<int>("output_csr_row_ptr.txt", 1);


    int V = csrRowPtr.size() - 1; // Number of vertices
    int s = 0; // Source
    int t = 1; // Sink

    // Convert vectors to pointers
    int *d_csrRowPtr = &csrRowPtr[0];
    int *d_colIndices = &colIndices[0];
    int *d_data = &data[0];

    int max_flow = fordFulkersonCuda(d_csrRowPtr, d_colIndices, d_data, s, t, V);
    std::cout << "The maximum possible flow is " << max_flow << std::endl;

    return 0;
}
