
// C++ program for implementation of Ford Fulkerson
// algorithm


// Number of vertices in given graph
// #define V 5

/* Returns true if there is a path from source 's' to sink
't' in residual graph. Also fills parent[] to store the
path */

/*
bool bfs(vector<vector<int>>& rGraph, int s, int t, vector<int>& parent)
{
	// Create a visited array and mark all vertices as not
	// visited
	bool visited[V];
	memset(visited, 0, sizeof(visited));

	// Create a queue, enqueue source vertex and mark source
	// vertex as visited
	queue<int> q;
	q.push(s);
	visited[s] = true;
	parent[s] = -1;

	// Standard BFS Loop
	while (!q.empty()) {
		int u = q.front();
		q.pop();

		for (int v = 0; v < V; v++) {
			if (visited[v] == false && rGraph[u][v] > 0) {
				// If we find a connection to the sink node,
				// then there is no point in BFS anymore We
				// just have to set its parent and can return
				// true
				if (v == t) {
					parent[v] = u;
					return true;
				}
				q.push(v);
				parent[v] = u;
				visited[v] = true;
			}
		}
	}

	// We didn't reach sink in BFS starting from source, so
	// return false
	return false;
}

// Returns the maximum flow from s to t in the given graph
int fordFulkerson(vector<vector<int>>& graph, int s, int t)
{
	int u, v;

	// Create a residual graph and fill the residual graph
	// with given capacities in the original graph as
	// residual capacities in residual graph
	vector<vector<int>> rGraph = graph; // Copy the graph to the residual graph
			 // Residual graph where rGraph[i][j]
				// indicates residual capacity of edge
				// from i to j (if there is an edge. If
				// rGraph[i][j] is 0, then there is not)
	for (u = 0; u < V; u++)
		for (v = 0; v < V; v++)
			rGraph[u][v] = graph[u][v];

	vector<int> parent(V); // Use a vector for the parent

	int max_flow = 0; // There is no flow initially

	// Augment the flow while there is path from source to
	// sink
	while (bfs(rGraph, s, t, parent)) {
		// Find minimum residual capacity of the edges along
		// the path filled by BFS. Or we can say find the
		// maximum flow through the path found.
		int path_flow = INT_MAX;
		for (v = t; v != s; v = parent[v]) {
			u = parent[v];
			path_flow = min(path_flow, rGraph[u][v]);
		}

		// update residual capacities of the edges and
		// reverse edges along the path
		for (v = t; v != s; v = parent[v]) {
			u = parent[v];
			rGraph[u][v] -= path_flow;
			rGraph[v][u] += path_flow;
		}

		// Add path flow to overall flow
		max_flow += path_flow;
	}

	// Return the overall flow
	return max_flow;
}

	


int main() {
    std::ifstream infile("data/1107data.mtx");
    std::vector<std::vector<int>> graph(V, std::vector<int>(V));

    for (int i = 0; i < V && infile; ++i) {
        for (int j = 0; j < V && infile; ++j) {
            infile >> graph[i][j];
        }
    }

    // Check if reading was successful
    if (!infile) {
        std::cerr << "Error reading from file!" << std::endl;
        return 1;
    }

    cout << "The maximum possible flow is "
		<< fordFulkerson(graph, 0, 2);

    return 0;
}*/

#include <iostream>
#include <vector>
#include <queue>
#include <climits> // for INT_MAX
#include <cmath>
#include <iostream>
#include <limits.h>
#include <queue>
#include <string.h>
#include <fstream>
#include <vector>
#include <iostream>
using namespace std;



bool bfs(const vector<int>& csrRowPtr, const vector<int>& colIndices, const vector<int>& capacity, int s, int t, vector<int>& parent) {
    int V = csrRowPtr.size() - 1;
    vector<bool> visited(V, false);
    queue<int> q;
    
    q.push(s);
    visited[s] = true;
    parent[s] = -1;

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int i = csrRowPtr[u]; i < csrRowPtr[u + 1]; i++) {
            int v = colIndices[i];
            if (!visited[v] && capacity[i] > 0) {
                if (v == t) {
                    parent[v] = u;
                    return true;
                }
                q.push(v);
                visited[v] = true;
                parent[v] = u;
            }
        }
    }
    
    return false;
}

int fordFulkerson(const vector<int>& csrRowPtr, const vector<int>& colIndices, vector<int> capacity, int s, int t) {
    int V = csrRowPtr.size() - 1;
    vector<int> parent(V);
    int max_flow = 0;

    while (bfs(csrRowPtr, colIndices, capacity, s, t, parent)) {
        int path_flow = INT_MAX;
        
        // Find minimum residual capacity of the augmenting path
        for (int v = t; v != s; v = parent[v]) {
            int u = parent[v];
            for (int i = csrRowPtr[u]; i < csrRowPtr[u + 1]; i++) {
                if (colIndices[i] == v) {
                    path_flow = min(path_flow, capacity[i]);
                    break;
                }
            }
        }
        
        // Update residual capacities of the edges and reverse edges
        for (int v = t; v != s; v = parent[v]) {
            int u = parent[v];
            for (int i = csrRowPtr[u]; i < csrRowPtr[u + 1]; i++) {
                if (colIndices[i] == v) {
                    capacity[i] -= path_flow;
                    break;
                }
            }
            // Note: This example does not explicitly handle reverse edges. You might need to adjust this for your use case.
        }

        max_flow += path_flow;
    }

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
    // Example graph in CSR format
    //vector<int> csrRowPtr = {0, 2, 4, 6, 8};
    //vector<int> colIndices = {1, 2, 0, 3, 1, 3, 0, 2};
    //vector<int> data = {1000, 1000, 1000, 1000, 1, 1000, 1, 1000};
	float scaleFactor = 1000.0f;

	std::vector<int> data = readVectorFromFile<int>("output_csr_data.txt", scaleFactor);
    std::vector<int> colIndices = readVectorFromFile<int>("output_csr_col_indices.txt", 1);
    std::vector<int> csrRowPtr = readVectorFromFile<int>("output_csr_row_ptr.txt", 1);


    int V = csrRowPtr.size()-1; // Number of vertices
    int s = 0; // Source
    int t = csrRowPtr.size()-2; // Sink

	/* // Example: Print the contents of csrRowPtr
    for (int i : data) {
        std::cout << i << " ";
    }
    std::cout << std::endl;

	for (int i : colIndices) {
        std::cout << i << " ";
    }
    std::cout << std::endl;

	for (int i : csrRowPtr) {
        std::cout << i << " ";
    }
    std::cout << std::endl; */

    int max_flow = fordFulkerson(csrRowPtr, colIndices, data, s, t);
    cout << "The maximum possible flow is " << max_flow << endl;

    return 0;
}
