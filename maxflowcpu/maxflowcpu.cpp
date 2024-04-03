#include <iostream>
#include <vector>
#include <queue>
#include <climits>
#include <fstream>
#include <sstream>
#include <string>
using namespace std;

#define INF 1e9

// Returns true if there is a path from source 's' to sink 't' in residual graph.
// Also fills parent[] to store the path.
bool bfs(vector<vector<double>>& rGraph, int s, int t, vector<int>& parent) {
    int V = rGraph.size();
    vector<bool> visited(V, false);
    queue<int> q;
    q.push(s);
    visited[s] = true;
    parent[s] = -1;

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int v = 0; v < V; v++) {
            if (!visited[v] && rGraph[u][v] > 0) {
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
    return false;
}

// Ford-Fulkerson algorithm
double fordFulkerson(vector<vector<double>>& graph, int s, int t) {
    int V = graph.size();
    vector<vector<double>> rGraph = graph;

    vector<int> parent(V, -1);
    double maxFlow = 0;

    while (bfs(rGraph, s, t, parent)) {
        double pathFlow = INF;
        for (int v = t; v != s; v = parent[v]) {
            int u = parent[v];
            pathFlow = min(pathFlow, rGraph[u][v]);
        }

        for (int v = t; v != s; v = parent[v]) {
            int u = parent[v];
            rGraph[u][v] -= pathFlow;
            rGraph[v][u] += pathFlow;
        }

        maxFlow += pathFlow;
    }

    return maxFlow;
}

int main() {
    ifstream infile("cage3.mtx");
    if (!infile) {
        cerr << "Error opening graph_data.mtx" << endl;
        return 1;
    }

    int numNodes;
    infile >> numNodes; // First line in mtx file contains number of nodes

    vector<vector<double>> graph(numNodes, vector<double>(numNodes, 0.0));

    int from, to;
    double weight;
    while (infile >> from >> to >> weight) {
        graph[from - 1][to - 1] = weight; // Adjust indices to start from 0
    }

    infile.close();

    int source = 0; // Source node
    int sink = 4;   // Sink node

    double maxFlow = fordFulkerson(graph, source, sink);
    cout << "The maximum possible flow is: " << maxFlow << endl;

    return 0;
}
