#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <queue>
#include <ctime> // For timing
using namespace std;

// Number of vertices in given graph
#define V 30000

#define INF 1e9

/* Returns true if there is a path from source 's' to sink
't' in residual graph. Also fills parent[] to store the
path */
bool bfs(vector<vector<int>>& rGraph, int s, int t, vector<int>& parent)
{
    // Create a visited array and mark all vertices as not
    // visited
    vector<bool> visited(V, false);

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

int fordFulkerson(vector<vector<int>>& graph, int s, int t)
{
    // Create a residual graph and fill the residual graph
    // with given capacities in the original graph as
    // residual capacities in residual graph
    vector<vector<int>> rGraph(V, vector<int>(V));
    for (int u = 0; u < V; u++)
    {
        for (int v = 0; v < V; v++)
        {
            rGraph[u][v] = graph[u][v];
        }
    }

    vector<int> parent(V); // This vector is filled by BFS and to
                           // store path

    int max_flow = 0; // There is no flow initially

    // Augment the flow while there is path from source to
    // sink
    while (bfs(rGraph, s, t, parent))
    {
        // Find minimum residual capacity of the edges along
        // the path filled by BFS. Or we can say find the
        // maximum flow through the path found.
        int path_flow = INT8_MAX;
        for (int v = t; v != s; v = parent[v])
        {
            int u = parent[v];
            path_flow = min(path_flow, rGraph[u][v]);
        }

        // update residual capacities of the edges and
        // reverse edges along the path
        clock_t start = clock(); // Start timing
        for (int v = t; v != s; v = parent[v])
        {
            int u = parent[v];
            rGraph[u][v] -= path_flow;
            rGraph[v][u] += path_flow;
        }
        clock_t end = clock(); // Stop timing
        double duration = double(end - start) / CLOCKS_PER_SEC;
        cout << "Time taken by augmenting paths update: " << duration << " seconds" << endl;

        // Add path flow to overall flow
        max_flow += path_flow;
    }

    // Return the overall flow
    return max_flow;
}

// Read input from .mtx file
void readInput(const char *filename, vector<vector<int>>& graph)
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

        source--;
        destination--;

        int scaledCapacity = static_cast<int>(capacity * 1000);
        graph[source][destination] = scaledCapacity;
    }

    file.close();
}

// Driver program to test above functions
int main()
{
    clock_t start = clock(); // Start timing

    // Let us create a graph shown in the above example
    vector<vector<int>> graph(V, vector<int>(V, 0));

    // Read the graph from .mtx file
    const char *filename = "/home/matthew.jezek/max-flow/main/30k200k";
    readInput(filename, graph);

    // Convert graph to rGraph
    // Let us consider the source is 0 and sink is V-1
    int source = 0, sink = V - 1;

    // Timing the fordFulkerson method
    int maxFlow = fordFulkerson(graph, source, sink);
    clock_t end = clock(); // Stop timing

    double duration = double(end - start) / CLOCKS_PER_SEC;
    cout << "Time taken by fordFulkerson: " << duration << " seconds" << endl;

    cout << "The maximum possible flow is " << maxFlow << endl;

    return 0;
}
