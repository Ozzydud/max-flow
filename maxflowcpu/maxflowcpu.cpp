#include <iostream>
#include <fstream>
#include <tuple>
#include <sstream>
#include <vector>
#include <queue>
#include <ctime> // For timing
using namespace std;


#define INF 1e9

// Timer function to measure time taken by a specific operation
double measureTime(clock_t start) {
    clock_t end = clock();
    return double(end - start) / CLOCKS_PER_SEC;
}

/* Returns true if there is a path from source 's' to sink
't' in residual graph. Also fills parent[] to store the
path */
pair<bool, double> bfs(vector<vector<int>>& rGraph, int s, int t, vector<int>& parent, int V)
{
    clock_t start = clock(); // Start timing

    double totalTime = 0.0;

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
                    totalTime += measureTime(start); // Accumulate time
                    return make_pair(true, totalTime);
                }
                q.push(v);
                parent[v] = u;
                visited[v] = true;
            }
        }
    }

    // We didn't reach sink in BFS starting from source, so
    // return false
    totalTime += measureTime(start); // Accumulate time
    return make_pair(false, totalTime);
}

tuple<int, double, double, double> fordFulkerson(vector<vector<int>>& graph, int s, int t, int V)
{
    double setupTime = 0;	
    clock_t start1 = clock();
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
    double duration = measureTime(start1); 
    setupTime += duration;
    vector<int> parent(V); // This vector is filled by BFS and to
                           // store path

    int max_flow = 0; // There is no flow initially

    double bfsTime = 0.0;
    double augmentingPathsTime = 0.0;

    // Augment the flow while there is path from source to
    // sink
    while (true)
    {
        pair<bool, double> result = bfs(rGraph, s, t, parent, V);
        bfsTime += result.second;
        if (!result.first)
            break;
	clock_t start = clock();
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
        for (int v = t; v != s; v = parent[v])
        {
            int u = parent[v];
            rGraph[u][v] -= path_flow;
            rGraph[v][u] += path_flow;
        }

        // Add path flow to overall flow
        max_flow += path_flow;
        augmentingPathsTime += measureTime(start); // End timing for augmenting paths
    }

   
    return make_tuple(max_flow, bfsTime, augmentingPathsTime, setupTime); // Return max flow and timings
}

// Read input from .mtx file
pair<vector<vector<int>>, double> readInput(const char *filename, int V)
{
    clock_t start = clock(); // Start timing

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

    vector<vector<int>> graph(V, vector<int>(V, 0));

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

    double duration = measureTime(start); // End timing
    return make_pair(graph, duration);
}

// Driver program to test above functions
int edmondskarp(const char *filename, int V)
{
    clock_t start = clock();	
    // Read the graph from .mtx file
    pair<vector<vector<int>>, double> readResult = readInput(filename, V);

    // Convert graph to rGraph
    // Let us consider the source is 0 and sink is V-1
    int source = 0, sink = V - 1;

    // Timing the fordFulkerson method
    tuple<int, double, double, double> fordFulkersonResult = fordFulkerson(readResult.first, source, sink, V);
    cout << "Init time: " << readResult.second + get<3>(fordFulkersonResult) << " seconds" << endl;
    cout << "Time taken by BFS: " << get<1>(fordFulkersonResult) << " seconds" << endl;
    cout << "Time taken by augmenting paths: " << get<2>(fordFulkersonResult) << " seconds" << endl;
    cout << "Total time: " << measureTime(start) << " seconds" <<endl; 
    cout << "The maximum possible flow is " << get<0>(fordFulkersonResult) << endl;

    return 0;
}

int main(){
    cout << "cage3" << endl; 
    edmondskarp("cage3.mtx", 5);
    cout << "cage3 end" << endl; 

    cout << "cage9" << endl; 
    edmondskarp("data/cage9.mtx", 3534);
    cout << "cage9 end" << endl; 

    cout << "cage10" << endl; 
    edmondskarp("data/cage10.mtx", 11397);
    cout << "cage10 end" << endl; 

    cout << "cage11" << endl; 
    edmondskarp("data/cage11.mtx", 39082);
    cout << "cage11 end" << endl; 
}
