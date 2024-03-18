#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <limits.h>
#include <queue>
#include <string.h>
#include <fstream>
#include <vector>
using namespace std;

#define V 6;

// Perform bfs to find augmenting paths
__global__ void CudaBfs(vector<vector<int>>& rGraph, int s, int t, vector<int>& parent){
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (*found || u >= V) return;

    if (visited[u]) {
        int start = u * V;
        for (int v = 0; v < V; ++v) {
            if (!visited[v] && rGraph[start + v] > 0) {
                parent[v] = u;
                if (v == t) {
                    *found = true;
                    return;
                }
                nextLevel[v] = 1;
            }
        }
    }
}

bool bfs(vector<vector<int>>& rGraph, int s, int t, vector<int>& parent){
    bool *visited;
    int visitedSize = V * sizeof(bool);


    cudaMalloc((void**) &visited, visitedSize);

    cudaMemset(visited, 0, visitedSize);

}


int main() {
    return 0;

}