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


void readInput(const char* filename, int total_nodes, int* residual) {

        ifstream file;
        file.open(filename);

        if (!file) {
        cout <<  "Error reading file!";
        exit(1);
    }

    string line;
    int source, destination;
    int numberOfEdges = 0;

    float capacity;
    cout << "before loop" << endl;
    while (getline(file, line)) {
        if (line.empty()) continue;

        stringstream linestream(line);
        //linestream >> source >> destination >> capacity;
        if (!(linestream >> source >> destination >> capacity)) {
    	cerr << "Error parsing line: " << line << endl;
   	 continue;
	}

	//cout << "reading lines" << endl;
        //cout << "Read: Source=" << source << ", Destination=" << destination << ", Capacity=" << capacity << endl;

        source--;
        destination--;
        //cout << "before scaling" << endl;
        int scaledCapacity = static_cast<int>(capacity * 1000);
        if (!residual) {
    	cerr << "Memory allocation failed for residual matrix.";
    	exit(EXIT_FAILURE);
	}

	numberOfEdges++;
	//cout << "after scaling" << endl;
        residual[source * total_nodes + destination] = scaledCapacity;
        //cout << "adding to residual" << endl;

        //cout << "Residual capacity[" << source << "][" << destination << "]: " << residual[source * total_nodes + destination] << endl;
        //counter++;
        //cout << counter << endl;
       
    }
    
    cout << "Number of edges in graph is: " << numberOfEdges << endl;
    file.close();
}

__global__ void cudaBFS(int *r_capacity, int *parent, int *flow, bool *frontier, bool* visited, int vertices, int sink, int* locks){
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (!frontier[sink] && Idx < vertices && frontier[Idx]) {
        frontier[Idx] = false;
        visited[Idx] = true;

        for (int i = vertices; i > Idx; i--) {
            if (!frontier[i] && !visited[i] && r_capacity[Idx * vertices + i] > 0) {
                if(atomicCAS(locks+i, 0 , 1) == 1 || frontier[i]){
                                continue;
                }
                frontier[i] = true;
                locks[i] = 0;


                parent[i] = Idx;
                flow[i] = min(flow[Idx], r_capacity[Idx * vertices + i]);
            }
        }

        for (int i = 0; i < Idx; i++) {
            if (!frontier[i] && !visited[i] && r_capacity[Idx * vertices + i] > 0) {
                if(atomicCAS(locks+i, 0 , 1) == 1 || frontier[i]){
                                continue;
                }
                frontier[i] = true;
                locks[i] = 0;
                parent[i] = Idx;
                flow[i] = min(flow[Idx], r_capacity[Idx * vertices + i]);
            }
        }
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
    int total_nodes = 39082; // Assuming 3534 or 1107 nodes or 11397 or 39082 or 130228
    int* residual;


    float avgBFSTime = 0;
    int bfsCounter = 0;
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);



    
    cudaEvent_t start, stop; // Declare start and stop events
    float milliseconds = 0; // Variable to store elapsed time in milliseconds

    // Initialize CUDA events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    float avgAUGTime = 0;
    int augCounter = 0;
    cudaEvent_t startEvent2, stopEvent2;
    cudaEventCreate(&startEvent2);
    cudaEventCreate(&stopEvent2);


    // Allocating memory for a square matrix representing the graph
    //residual = (int*)malloc(sizeof(int) * total_nodes * total_nodes);
    cout << "residual" << endl;
    //memset(residual, 0, sizeof(int) * total_nodes * total_nodes);
    cout << "residual1" << endl;

    try {
	residual = new int[total_nodes * total_nodes]();
    } catch (const std::bad_alloc& e) {
	    std::cerr << "Failed to allocate memory for the residual matrix: " << e.what() << std::endl;
	    return 1;
    }





    readInput("data/cage11.mtx", total_nodes, residual);
    cout << "data read" << endl;

    int source = 0;
    int sink = total_nodes - 1; // Assuming sink is the last node
    int path_flow;

    int* parent = new int[total_nodes];
    int* flow = new int[total_nodes];
    bool* frontier = new bool[total_nodes];
    bool* visited = new bool [total_nodes];
    bool* do_change_capacity = new bool[total_nodes];
    

    // Set initial flow from source to 0
    flow[source] = 0;
    int* locks = new int[total_nodes];
    int* d_r_capacity, * d_parent, * d_flow, *d_locks;;
    bool* d_frontier, * d_visited, *d_do_change_capacity;

    size_t locks_size = total_nodes * sizeof(int);
    
    cout << "hi1" << endl;
    // Allocate memory on device
    cudaMalloc((void**)&d_r_capacity, total_nodes * total_nodes * sizeof(int));
    cout << "hi1" << endl;
    cudaMalloc((void**)&d_parent, total_nodes * sizeof(int));
    cout << "hi2" << endl;
    cudaMalloc((void**)&d_flow, total_nodes * sizeof(int));
    cout << "hi3" << endl;
    cudaMalloc((void**)&d_frontier, total_nodes * sizeof(bool));
    cout << "hi4" << endl;
    cudaMalloc((void**)&d_visited, total_nodes * sizeof(bool));
    cout << "hi5" << endl;
    cudaMalloc((void**)&d_do_change_capacity, total_nodes * sizeof(bool));
    cout << "hi6" << endl;
    cudaMalloc((void**)&d_locks, locks_size);


    // Copy data from host to device
    cudaMemcpy(d_r_capacity, residual, total_nodes * total_nodes * sizeof(int), cudaMemcpyHostToDevice);


    bool found_augmenting_path;
    int max_flow = 0;
    int block_size = 1024;
    int grid_size = ceil(total_nodes * 1.0 / block_size); //(total_nodes + block_size - 1) / block_size;

    
    cout << "hi1" << endl;
    int counter = 0;

    do{
        for (int i = 0; i < total_nodes; ++i) {
        parent[i] = -1; // Initialize parent array
        flow[i] = INF;  // Initialize flow array with INF
        locks[i] = 0;
        if(i == source){
            frontier[i] = true;
        }else{
            frontier[i] = false;
        }

        visited[i] = false;
        do_change_capacity[i] = false;
        }
   
        cudaMemcpy(d_parent, parent, total_nodes * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_flow, flow, total_nodes * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_frontier, frontier, total_nodes * sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(d_visited, visited, total_nodes * sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(d_locks, locks, locks_size, cudaMemcpyHostToDevice);
	    //cout << "hi2" << endl;
        while(!sink_reachable(frontier, total_nodes, sink)){
        cudaEventRecord(startEvent, 0);

        // Run BFS kernel
        cudaBFS<<<grid_size, block_size>>>(d_r_capacity, d_parent, d_flow, d_frontier, d_visited, total_nodes, sink, d_locks);
        bfsCounter++;
        // Stop recording the event
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);

        // Calculate elapsed time
        float bfsmili = 0.0f;
        cudaEventElapsedTime(&bfsmili, startEvent, stopEvent);
        avgBFSTime += bfsmili;
        

        cudaMemcpy(frontier, d_frontier, total_nodes * sizeof(bool), cudaMemcpyDeviceToHost);
        }

        found_augmenting_path = frontier[sink];

        if(!found_augmenting_path){
            break;
        }

        cudaMemcpy(flow, d_flow, total_nodes * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(parent, d_parent, total_nodes * sizeof(int), cudaMemcpyDeviceToHost);

        path_flow = flow[sink];
        max_flow += path_flow;

        for(int i = sink; i != source; i = parent[i]){
                        do_change_capacity[i] = true;
                }

        cudaMemcpy(d_do_change_capacity, do_change_capacity, total_nodes * sizeof(bool), cudaMemcpyHostToDevice);

	//cout << "hi4" << endl;
        // Launch BFS kernel
    cudaEventRecord(startEvent2, 0);
    for(int i = sink; i != source; i = parent[i]){
        if(do_change_capacity[i]){
        residual[parent[i] * total_nodes + i] -= path_flow;
        residual[i * total_nodes + parent[i]] += path_flow; 
        }
    }
    
    cudaMemcpy(d_r_capacity, residual, total_nodes * total_nodes * sizeof(int), cudaMemcpyHostToDevice);
        augCounter++;
        // Stop recording the event
        cudaEventRecord(stopEvent2, 0);
        cudaEventSynchronize(stopEvent2);

        // Calculate elapsed time
        float augmili = 0.0f;
        cudaEventElapsedTime(&augmili, startEvent2, stopEvent2);
        avgAUGTime += augmili;
	//cout << path_flow << endl;
	counter++;
	//cout << "Counter is: " << counter << endl;

    } while(found_augmenting_path); //found_augmenting_path);
    cout << "Counter is: " << counter << endl;
    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Time for BFS and augmenting path: " << milliseconds << " ms\n";
    cout << "Average BFS time is: " << avgBFSTime / bfsCounter << "ms\n";
    cout << "Total time BFS is: " << avgBFSTime << "ms\n";
    cout << "Average AUG time is " << avgAUGTime << "ms\n";
    cout << "Total AUG time is: " << avgAUGTime / augCounter << "ms\n";

    cout << "Maximum Flow: " << max_flow << endl;
    

    // Clean up allocated memory
    delete[] residual;
    delete[] parent;
    delete[] flow;
    delete[] locks;
    delete[] frontier;
    delete[] visited;
    delete[] do_change_capacity;
    cudaFree(d_r_capacity);
    cudaFree(d_parent);
    cudaFree(d_flow);
    cudaFree(d_frontier);
    cudaFree(d_visited);
    cudaFree(d_locks);
    cudaFree(d_do_change_capacity);
    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(stopEvent);
    cudaEventDestroy(startEvent);

    cudaEventDestroy(stopEvent2);
    cudaEventDestroy(startEvent2);

    return 0;
}


