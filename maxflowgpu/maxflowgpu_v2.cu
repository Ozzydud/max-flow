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
#include <cuda_runtime_api.h>

using namespace std;

#define INF 1e9


class Edge{
	public:
		int from;
		int to;
		float weight;	
	};


Edge *readInput(const char* filename, int total_nodes, int num_edges){
	FILE* file = fopen(filename, "r");

   	if (!file) {
        std::cout << "Error reading file!";
        exit(1);
    }

    	// Allocate memory for storing data on the host
    	Edge *edges = (Edge *)malloc(sizeof(Edge) * num_edges);

    	int from, to;
    	float weight;
	

    	// Read the data from the file into the edges array
    	for (int i = 0; i < num_edges; i++){
        	fscanf(file, "%d %d %f", &from, &to, &weight);
		int scaledweight = static_cast<int>(weight*1000);
        	edges[i] = {from - 1, to - 1, scaledweight};
		//cout << edges[0].weight << endl;
    }
	fclose(file);
    	return edges;
}


__global__ void cudaBFS(Edge *edges, bool *frontier, bool* visited, int vertices, int sink, int* locks){
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (!frontier[sink] && Idx < vertices && frontier[Idx]) {
        frontier[Idx] = false;
        visited[Idx] = true;

        for (int i = Idx; i < vertices; i++) {
            if (!frontier[i] && !visited[i] && edges[i].weight > 0) {
                if(atomicCAS(locks+i, 0 , 1) == 1 || frontier[i]){
                                continue;
                }
                frontier[i] = true;
                locks[i] = 0;


                edges[i].from = Idx;
                edges[i].weight = min(edges[Idx].weight, edges[i].weight);
            }
        }

        for (int i = 0; i < Idx; i++) {
            if (!frontier[i] && !visited[i] && edges[i].weight > 0) {
                if(atomicCAS(locks+i, 0 , 1) == 1 || frontier[i]){
                                continue;
                }
                frontier[i] = true;
                locks[i] = 0;
                edges[i].from = Idx;
                edges[i].weight = min(edges[Idx].weight, edges[i].weight);
            }
        }
    }
}


__global__ void cudaAugment_path(Edge *edges, bool* do_change_capacity, int total_nodes, int path_flow){
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(Idx < total_nodes && do_change_capacity[Idx]){

	edges[Idx].weight -= path_flow;
	//maybe to instead of from?
	edges[Idx].weight += path_flow; 
    }    
}


bool sink_reachable(bool* frontier, int total_nodes, int sink){
    for (int i = total_nodes-1; i > -1; --i) {
                if(frontier[i]){
                        return i == sink;
                }
        }
        return true;
}



int main() {
    cudaError_t cudaStatus = cudaSetDevice(4);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?";
        return 1;
    }
    int total_nodes = 5; // Assuming 3534 or 1107 nodes or 11397 or 39082 or 130228
    int num_edges = 19;
    Edge* edges;
    
    cudaEvent_t start, stop; // Declare start and stop events
    float milliseconds = 0; // Variable to store elapsed time in milliseconds

    // Initialize CUDA events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


   edges = readInput("cage3.mtx", total_nodes, num_edges);
  

    int source = 0;
    int sink = total_nodes - 1; // Assuming sink is the last node
    int path_flow;

    
  
    bool* frontier = new bool[total_nodes];
    bool* visited = new bool [total_nodes];
    bool* do_change_capacity = new bool[total_nodes];
    

    // Set initial flow from source to 0
    edges[source].weight = 0;
    int* locks = new int[total_nodes];
    Edge* d_edges;
    int *d_locks;
    bool* d_frontier, * d_visited, *d_do_change_capacity;

    size_t locks_size = total_nodes * sizeof(int);
    
    // Allocate memory on device
    cudaMalloc((void**)&d_edges, num_edges * sizeof(Edge));
    cudaMalloc((void**)&d_frontier, total_nodes * sizeof(bool));
    cudaMalloc((void**)&d_visited, total_nodes * sizeof(bool));
    cudaMalloc((void**)&d_do_change_capacity, total_nodes * sizeof(bool));
    cudaMalloc((void**)&d_locks, locks_size);


    // Copy data from host to device
    cudaMemcpy(d_edges, edges, num_edges * sizeof(Edge), cudaMemcpyHostToDevice);


    bool found_augmenting_path;
    int max_flow = 0;
    int block_size = 1024;
    int grid_size = ceil(total_nodes * 1.0 / block_size); //(total_nodes + block_size - 1) / block_size;

    
    cout << "hi1" << endl;
    int counter = 0;

    do{
	cudaMemcpy(d_edges, edges, num_edges * sizeof(Edge), cudaMemcpyHostToDevice);
	cout << edges[1].weight << edges[1].from << edges[1].to << endl;
        for (int i = 0; i < total_nodes; ++i) {
        edges[i].weight = INF; // Initialize flow array with INF
        locks[i] = 0;
        if(i == source){
            frontier[i] = true;
        }else{
            frontier[i] = false;
        }

	//cout<< frontier[i] << endl;

        visited[i] = false;
        do_change_capacity[i] = false;
        }
   	
        cudaMemcpy(d_frontier, frontier, total_nodes * sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(d_visited, visited, total_nodes * sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(d_locks, locks, locks_size, cudaMemcpyHostToDevice);
	cout << "hi2" << endl;
        while(!sink_reachable(frontier, total_nodes, sink)){
        cudaBFS<<<grid_size, block_size>>>(d_edges, d_frontier, d_visited, total_nodes, sink, d_locks);
        cout << "hi3" << endl;
        

        cudaMemcpy(frontier, d_frontier, total_nodes * sizeof(bool), cudaMemcpyDeviceToHost);
	//for(int i = 0; i < total_nodes; i++){
	//	cout << frontier[i] << endl;
	//}
        }

        found_augmenting_path = frontier[sink];
	cout << found_augmenting_path << endl;
        if(!found_augmenting_path){
            break;
        }

	cout << edges[0].weight << endl;

	cudaMemcpy(edges, d_edges, num_edges * sizeof(Edge), cudaMemcpyDeviceToHost);
	
        path_flow = edges[num_edges - 1].weight;
	cout << "path flow"  << path_flow << endl;
        max_flow += path_flow;

        for(int i = sink; i != source; i = edges[i].from){
                        do_change_capacity[i] = true;
                }

        cudaMemcpy(d_do_change_capacity, do_change_capacity, total_nodes * sizeof(bool), cudaMemcpyHostToDevice);

	cout << "hi4" << endl;
        // Launch BFS kernel
        cudaAugment_path<<< grid_size, block_size >>>(d_edges, d_do_change_capacity, total_nodes, path_flow);
	cout << path_flow << endl;
	counter++;


    } while(found_augmenting_path); //found_augmenting_path);
    cout << "hi6" << endl;
    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Time for BFS and augmenting path: " << milliseconds << " ms\n";

    cout << "Maximum Flow: " << max_flow << endl;
    

    // Clean up allocated memory
    delete[] edges;
    delete[] locks;
    delete[] frontier;
    delete[] visited;
    delete[] do_change_capacity;
    cudaFree(d_edges);
    cudaFree(d_frontier);
    cudaFree(d_visited);
    cudaFree(d_locks);
    cudaFree(d_do_change_capacity);
    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}


