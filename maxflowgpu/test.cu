#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

using namespace std;

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>

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

        residual[source * total_nodes + destination] = scaledCapacity;
        numberOfEdges++;
    }

    cout << "Number of edges in graph is: " << numberOfEdges << endl;
    file.close();
}



__global__ void top_down_step(int *frontier, int frontier_size, bool *visited, int *residual, int total_nodes, int *next_frontier, int *next_frontier_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < frontier_size) {
        int node = frontier[idx];
        for (int neighbor = 0; neighbor < total_nodes; ++neighbor) {
            if (residual[node * total_nodes + neighbor] > 0 && !visited[neighbor]) {
                if (atomicExch(&visited[neighbor], true) == false) {
                    int pos = atomicAdd(next_frontier_size, 1);
                    next_frontier[pos] = neighbor;
                }
            }
        }
    }
}

__global__ void bottom_up_step(int *frontier, int frontier_size, bool *visited, int *residual, int total_nodes, int *next_frontier, int *next_frontier_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_nodes && !visited[idx]) {
        for (int j = 0; j < total_nodes; ++j) {
            if (residual[j * total_nodes + idx] > 0 && visited[j]) {
                if (atomicExch(&visited[idx], true) == false) {
                    int pos = atomicAdd(next_frontier_size, 1);
                    next_frontier[pos] = idx;
                    break;
                }
            }
        }
    }
}




void bfs(int *residual, int total_nodes, int start_node) {
    const int BLOCK_SIZE = 256;

    int *d_residual;
    cudaMalloc(&d_residual, sizeof(int) * total_nodes * total_nodes);
    cudaMemcpy(d_residual, residual, sizeof(int) * total_nodes * total_nodes, cudaMemcpyHostToDevice);

    bool *d_visited;
    int *d_frontier, *d_next_frontier, *d_frontier_size, *d_next_frontier_size;
    cudaMalloc(&d_visited, sizeof(bool) * total_nodes);
    cudaMalloc(&d_frontier, sizeof(int) * total_nodes);
    cudaMalloc(&d_next_frontier, sizeof(int) * total_nodes);
    cudaMalloc(&d_frontier_size, sizeof(int));
    cudaMalloc(&d_next_frontier_size, sizeof(int));

    int h_frontier_size = 1;
    cudaMemcpy(d_frontier, &start_node, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_frontier_size, &h_frontier_size, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_visited, 0, sizeof(bool) * total_nodes);
    cudaMemset(d_visited + start_node, 1, sizeof(bool));

    bool top_down = true;
    while (h_frontier_size > 0) {
        cudaMemset(d_next_frontier_size, 0, sizeof(int));

        if (top_down) {
            int num_blocks = (h_frontier_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            top_down_step<<<num_blocks, BLOCK_SIZE>>>(d_frontier, h_frontier_size, d_visited, d_residual, total_nodes, d_next_frontier, d_next_frontier_size);
        } else {
            int num_blocks = (total_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
            bottom_up_step<<<num_blocks, BLOCK_SIZE>>>(d_frontier, h_frontier_size, d_visited, d_residual, total_nodes, d_next_frontier, d_next_frontier_size);
        }

        cudaMemcpy(&h_frontier_size, d_next_frontier_size, sizeof(int), cudaMemcpyDeviceToHost);

        std::swap(d_frontier, d_next_frontier);
        cudaMemcpy(d_frontier_size, d_next_frontier_size, sizeof(int), cudaMemcpyDeviceToDevice);

        top_down = h_frontier_size < total_nodes / 10;  // Example threshold to switch strategies
    }

    // Free allocated memory
    cudaFree(d_residual);
    cudaFree(d_visited);
    cudaFree(d_frontier);
    cudaFree(d_next_frontier);
    cudaFree(d_frontier_size);
    cudaFree(d_next_frontier_size);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <input_file> <total_nodes>" << endl;
        return 1;
    }

    const char *filename = argv[1];
    int total_nodes = atoi(argv[2]);

    int *residual = new int[total_nodes * total_nodes]();
    readInput(filename, total_nodes, residual);

    int start_node = 0; // Assuming the BFS starts from node 0
    bfs(residual, total_nodes, start_node);

    delete[] residual;
    return 0;
}
