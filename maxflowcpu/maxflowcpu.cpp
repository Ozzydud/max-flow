#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <limits.h>
#include <queue>
#include <string.h>
using namespace std;

// Number of vertices in given graph
#define V 5

/* Returns true if there is a path from source 's' to sink
't' in residual graph. Also fills parent[] to store the
path */
bool bfs(int rGraph[V][V], int s, int t, int parent[])
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
    while (!q.empty())
    {
        int u = q.front();
        q.pop();

        for (int v = 0; v < V; v++)
        {
            if (visited[v] == false && rGraph[u][v] > 0)
            {
                // If we find a connection to the sink node,
                // then there is no point in BFS anymore We
                // just have to set its parent and can return
                // true
                if (v == t)
                {
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
int fordFulkerson(int graph[V][V], int s, int t)
{
    int u, v;

    // Create a residual graph and fill the residual graph
    // with given capacities in the original graph as
    // residual capacities in residual graph
    int rGraph[V][V]; // Residual graph where rGraph[i][j]
                      // indicates residual capacity of edge
                      // from i to j (if there is an edge. If
                      // rGraph[i][j] is 0, then there is not)
    for (u = 0; u < V; u++)
        for (v = 0; v < V; v++)
            rGraph[u][v] = graph[u][v];

    int parent[V]; // This array is filled by BFS and to
                   // store path

    int max_flow = 0; // There is no flow initially

    // Augment the flow while there is path from source to
    // sink
    while (bfs(rGraph, s, t, parent))
    {
        // Find minimum residual capacity of the edges along
        // the path filled by BFS. Or we can say find the
        // maximum flow through the path found.
        int path_flow = INT_MAX;
        for (v = t; v != s; v = parent[v])
        {
            u = parent[v];
            path_flow = min(path_flow, rGraph[u][v]);
        }

        // update residual capacities of the edges and
        // reverse edges along the path
        for (v = t; v != s; v = parent[v])
        {
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

// Read input from .mtx file
void readInput(const char* filename, int total_nodes, int graph[V][V])
{
    ifstream file;
    file.open(filename);

    if (!file)
    {
        cout << "Error reading file!";
        exit(1);
    }

    string line;
    int source, destination;
    float capacity;

    while (getline(file, line))
    {
        if (line.empty())
            continue;

        stringstream linestream(line);
        linestream >> source >> destination >> capacity;

        // cout << "Read: Source=" << source << ", Destination=" << destination << ", Capacity=" << capacity << endl;

        source--;
        destination--;

        int scaledCapacity = static_cast<int>(capacity * 1000);
        graph[source][destination] = scaledCapacity;

        // cout << "Graph[" << source << "][" << destination << "]: " << graph[source][destination] << endl;
    }

    file.close();
}

// Driver program to test above functions
int main()
{
    // Let us create a graph shown in the above exampl
    int graph[V][V] = {};

    // Read the graph from .mtx file
    const char* filename = "cage3.mtx";
    int total_nodes = V;
    readInput(filename, total_nodes, graph);

    // Let us consider the source is 0 and sink is 4
    int source = 0, sink = 4;

    cout << "The maximum possible flow is " << fordFulkerson(graph, source, sink) << endl;

    return 0;
}