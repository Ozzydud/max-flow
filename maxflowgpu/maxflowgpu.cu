#include <iostream>
#include <fstream>
#include <vector>

// CUDA libraries
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define INF 1e9

// BFS
__global__ void cudaBFS (int *row, int *indices, int *data,
                         int source, int sink, int *parent, int *queue, int *flow, int *residual, bool *visited){
    int tid = blockIdx.x * blockDim.x * threadIdx.x; //Finding thread ID
    int vertices = 5;
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
                            int source, int sink){
    int tid = blockIdx.x * blockDim.x * threadIdx.x; //Finding thread ID
    int vertices = 5;
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


int main() {
    std::cout << "no errors plz";
    return 0;

}
