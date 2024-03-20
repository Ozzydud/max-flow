#include <iostream>
#include <fstream>
#include <vector>

// CUDA libraries
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void cudaBFS (int *row, int *indices, int *data,
                         int source, int sink, int *parent, int *queue, int *flow, int *residual, bool *visited){
    int tid = blockIdx.x * blockDim.x * threadIdx.x; //Finding thread ID
    int vertices = row.size()-1;

    if(visited[tid] == false && vertices > tid){ //Mark as visited and add tid to the queue
        queue[tid] = tid;
        visited[tid] = true;
        parent[tid] = -1;
    }

     __syncthreads(); // Not optimal - we need to wait for all threads before we do BFS

     while (!visited[sink] && !visited[source]) { //We keep going as long as we have not visited both sink and source
        for(int v = 0; v<vertices; v++){
            if(!visited[v] && residual[tid * vertices + v] > 0){
                // Process neighboring vertices
                    queue[v] = tid;
                    visited[v] = true;
                    parent[v] = tid;
            }
        }
         __syncthreads();
     }

}


int main() {
    std::cout << "no errors plz";
    return 0;

}
