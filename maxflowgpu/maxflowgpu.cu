#include <iostream>
#include <vector>
#include <queue>
#include <cuda_runtime.h>

#define V 6 // Assuming V is a constant

__global__ void cudaBFS(int *rGraph, bool *visited, int *parent, int *nextLevel, bool *found, int t) {
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

bool bfs(int rGraph[V][V], int s, int t, int parent[V]) {
    int *dev_rGraph, *dev_parent, *dev_nextLevel;
    bool *dev_visited, *dev_found;
    bool found = false;

    int rGraphSize = V * V * sizeof(int);
    int visitedSize = V * sizeof(bool);
    int parentSize = V * sizeof(int);

    cudaMalloc((void**)&dev_rGraph, rGraphSize);
    cudaMalloc((void**)&dev_visited, visitedSize);
    cudaMalloc((void**)&dev_parent, parentSize);
    cudaMalloc((void**)&dev_nextLevel, visitedSize);
    cudaMalloc((void**)&dev_found, sizeof(bool));

    cudaMemcpy(dev_rGraph, rGraph, rGraphSize, cudaMemcpyHostToDevice);
    cudaMemset(dev_visited, 0, visitedSize);
    cudaMemset(dev_nextLevel, 0, visitedSize);
    cudaMemset(dev_found, 0, sizeof(bool));

    std::queue<int> q;
    q.push(s);
    cudaMemcpy(dev_parent + s, &s, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_visited + s, &s, sizeof(bool), cudaMemcpyHostToDevice);

    while (!q.empty() && !found) {
        int u = q.front();
        q.pop();
        cudaBFS<<<(V + 255) / 256, 256>>>(dev_rGraph, dev_visited, dev_parent, dev_nextLevel, dev_found, t);
        cudaDeviceSynchronize();

        cudaMemcpy(&found, dev_found, sizeof(bool), cudaMemcpyDeviceToHost);
        if (found) break;

        cudaMemcpy(dev_visited, dev_nextLevel, visitedSize, cudaMemcpyDeviceToDevice);
        cudaMemset(dev_nextLevel, 0, visitedSize);

        for (int v = 0; v < V; ++v) {
            if (rGraph[u][v] > 0 && !dev_visited[v]) {
                q.push(v);
                cudaMemcpy(dev_parent + v, &u, sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(dev_visited + v, &v, sizeof(bool), cudaMemcpyHostToDevice);
            }
        }
    }

    if (found) {
        int current = t;
        while (current != -1) {
            parent[current] = dev_parent[current];
            cudaMemcpy(&current, dev_parent + current, sizeof(int), cudaMemcpyDeviceToHost);
        }
    }

    cudaFree(dev_rGraph);
    cudaFree(dev_visited);
    cudaFree(dev_parent);
    cudaFree(dev_nextLevel);
    cudaFree(dev_found);

    return found;
}

int main() {
    int rGraph[V][V] = {
        {0, 16, 13, 0, 0, 0},
        {0, 0, 10, 12, 0, 0},
        {0, 4, 0, 0, 14, 0},
        {0, 0, 9, 0, 0, 20},
        {0, 0, 0, 7, 0, 4},
        {0, 0, 0, 0, 0, 0}
    };

    int parent[V];
    int s = 0, t = 5;
    if (bfs(rGraph, s, t, parent)) {
        std::cout << "Path found from " << s << " to " << t << ":\n";
        for (int i = 0; i < V; ++i) {
            std::cout << i << " <- ";
            if (parent[i] == -1) std::cout << "Source";
            else std::cout << parent[i];
            std::cout << "\n";
        }
    } else {
        std::cout << "No path exists from " << s << " to " << t << "\n";
    }

    return 0;
}
