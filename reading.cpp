#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <string>
#include <queue>
#include <set>
#include <unordered_map>
#include <algorithm>



using namespace std;

const int INF = 1e9;
int global_root = 0;

int delta = 4; // TODO fine-tune

struct Edge {
    int v1, v2;
    long long weight;
};

struct Vertex {
    int id;
    vector<Edge> edges;
};

unordered_map<int, long long> delta_stepping(unordered_map<int, Vertex> vertex_mapping, int root) {
    unordered_map<int, long long> d;
    unordered_map<long long, set<int>> buckets;


    set<int> zero_set;
    set<int> inf_set;

    buckets[INF] = zero_set;
    for (auto& it: vertex_mapping) {
        if (root == root) {
            d[v.id] = 0;
            buckets[0] = zero_set;
            buckets[0].insert(it.first);
        }
        else {
            d[it.first] = INF;
        }
    }


    for (int k = 0; k < INF; k++) {

        if (buckets.count(k) == 0) {
            continue;
        }

        set<int> A = buckets[k];

        // Process bucket
        while(!A.empty()){
            set<int> A_prim;
            for (int u: A) {
                Vertex current_vertex = vertex_mapping[u];
                vector<Edge> edges = current_vertex.edges;

                for (Edge e: edges) {
                    // Relax edge
                    int u = e.v1;
                    int v = e.v2;
                    long long w = e.weight;

                    int old_bucket = d[v]/delta;
                    long long old_d = d[v];
                    d[v] = min(d[v], d[u] + w);

                    if (d[u] + w < d[v]) {
                        long long old_bucket = d[v] / delta;
                        d[v] = d[u] + w;
                        long long new_bucket = d[v] / delta;

                        if(new_bucket < old_bucket) {
                            buckets[old_bucket].erase(v);

                            if (buckets.count(new_bucket) == 0) {
                                set<int> new_set;
                                buckets[new_bucket] = new_set;
                            }
                            buckets[new_bucket].insert(v);
                        }

                        A_prim.insert(v);

                    }
                    
                }
            }

            A.clear();

            set_intersection(A_prim.begin(), A_prim.end(),
                          buckets[k].begin(), buckets[k].end(),
                          inserter(A, A.begin()));
        }

        buckets[k].clear();
    }

    return d;
}






int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 3) {
        if (rank == 0) {
            std::cerr << "Usage: ./sssp <input_file> <output_file>\n";
        }
        MPI_Finalize();
        return 1;
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];

    std::ifstream infile(input_file);
    if (!infile.is_open()) {
        std::cerr << "Rank " << rank << ": Failed to open input file " << input_file << "\n";
        MPI_Finalize();
        return 1;
    }

    int num_vertices, start_vertex, end_vertex;
    infile >> num_vertices >> start_vertex >> end_vertex;

    unordered_map<int, Vertex> my_vertices;

    for (int i = start_vertex, i <= end_vertex; i++) {
        Vertex v;
        v.id = i;
        vector<long long> edges;
        v.edges = edges;
        my_vertices[i] = v;
    }

    int u;
    int v;
    long long w;

    while (infile >> u >> v >> w) {

        if (u >= start_vertex && u <= end_vertex) {
            Edge e;
            e.v1 = u;
            e.v2 = v;
            e.weight = w;
            my_vertices[u].edges.push_back(e);
        }

        if (v >= start_vertex && v <= end_vertex) {
            Edge e;
            e.v1 = v;
            e.v2 = u;
            e.weight = w;
            my_vertices[u].edges.push_back(e);
        }    

    }
    infile.close();

    unordered_map<int, long long> final_values = delta_stepping(my_vertices, global_root);

    std::cout << "Rank " << rank << " has vertices " << start_vertex << " to " << end_vertex 
              << " and read " << edges.size() << " edges.\n";

    // Dummy output for testing (write -1 as shortest path for each vertex)
    std::ofstream outfile(output_file);
    if (!outfile.is_open()) {
        std::cerr << "Rank " << rank << ": Failed to open output file " << output_file << "\n";
        MPI_Finalize();
        return 1;
    }

    for (int v = start_vertex; v <= end_vertex; ++v) {
        outfile << final_values[v] << "\n";
    }
    outfile.close();

    MPI_Finalize();
    return 0;
}
