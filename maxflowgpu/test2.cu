__global__ void bfs(int *residual, int *parent, int *visited, int total_nodes, int source, int sink) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_nodes) return;

    __shared__ int queue[1024];
    __shared__ int front, rear;

    if (idx == 0) {
        queue[0] = source;
        front = 0;
        rear = 1;
    }
    __syncthreads();

    while (front < rear) {
        int current_node = queue[front++];
        __syncthreads();

        for (int i = 0; i < total_nodes; ++i) {
            if (residual[current_node * total_nodes + i] > 0 && !visited[i]) {
                parent[i] = current_node;
                visited[i] = 1;
                if (i == sink) return;
                queue[rear++] = i;
            }
        }
        __syncthreads();
    }
}


__global__ void updateResidual(int *residual, int *parent, int total_nodes, int source, int sink, int path_flow) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_nodes) return;

    int v = sink;
    while (v != source) {
        int u = parent[v];
        atomicSub(&residual[u * total_nodes + v], path_flow);
        atomicAdd(&residual[v * total_nodes + u], path_flow);
        v = u;
    }
}

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cuda_runtime.h>

using namespace std;

void readInput(const char* filename, int total_nodes, int* residual);

__global__ void bfs(int *residual, int *parent, int *visited, int total_nodes, int source, int sink);
__global__ void updateResidual(int *residual, int *parent, int total_nodes, int source, int sink, int path_flow);

int main() {
    const char* filename = "cage3.mtx"; // Path to your input file
    int total_nodes = ; // Number of nodes in your graph, set appropriately
    int source = 0, sink = total_nodes - 1;

    int *residual = (int *)malloc(total_nodes * total_nodes * sizeof(int));
    readInput(filename, total_nodes, residual);

    int *d_residual, *d_parent, *d_visited;
    cudaMalloc((void**)&d_residual, total_nodes * total_nodes * sizeof(int));
    cudaMalloc((void**)&d_parent, total_nodes * sizeof(int));
    cudaMalloc((void**)&d_visited, total_nodes * sizeof(int));

    cudaMemcpy(d_residual, residual, total_nodes * total_nodes * sizeof(int), cudaMemcpyHostToDevice);

    int max_flow = 0;

    do {
        cudaMemset(d_parent, -1, total_nodes * sizeof(int));
        cudaMemset(d_visited, 0, total_nodes * sizeof(int));

        int num_blocks = (total_nodes + 1023) / 1024;
        bfs<<<num_blocks, 1024>>>(d_residual, d_parent, d_visited, total_nodes, source, sink);
        cudaDeviceSynchronize();

        vector<int> parent(total_nodes);
        cudaMemcpy(parent.data(), d_parent, total_nodes * sizeof(int), cudaMemcpyDeviceToHost);

        if (parent[sink] == -1) break; // No path found

        int path_flow = INT_MAX;
        for (int v = sink; v != source; v = parent[v]) {
            int u = parent[v];
            path_flow = min(path_flow, residual[u * total_nodes + v]);
        }

        updateResidual<<<num_blocks, 1024>>>(d_residual, d_parent, total_nodes, source, sink, path_flow);
        cudaDeviceSynchronize();

        max_flow += path_flow;

    } while (true);

    cout << "Maximum flow: " << max_flow << endl;

    free(residual);
    cudaFree(d_residual);
    cudaFree(d_parent);
    cudaFree(d_visited);

    return 0;
}

