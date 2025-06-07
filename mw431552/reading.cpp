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

int owner(int v, int num_procs) {
    return v/num_procs;
}

int local_index(int v, int num_procs) {
    return v % num_procs;
}

unordered_map<int, long long> delta_stepping(unordered_map<int, Vertex> vertex_mapping, int root, int rank, int num_procs, int local_vertex_count) {    
    vector<long long> local_d(local_vertex_count, INF);
    vector<long long> local_changed(local_vertex_count, 0);
    vector<long long> local_d_prev(local_vertex_count, INF);

    // Setup MPI Windows
    MPI_Win win_d, win_changed;

    MPI_Win_create(local_d.data(), local_vertex_count * sizeof(long long),
                   sizeof(long long), MPI_INFO_NULL, MPI_COMM_WORLD, &win_d);

    MPI_Win_create(local_changed.data(), local_vertex_count * sizeof(int),
                   sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_changed);

    unordered_map<long long, set<int>> buckets;

    if (owner(root, num_procs) == rank) {
        local_d[local_index(root, num_procs)] = 0;
        local_d_prev[local_index(root, num_procs)] = 0;
        buckets[0].insert(root);
    }

    cout << "MY RANK IS: " << rank << "\n";


    MPI_Barrier(MPI_COMM_WORLD); // Ensure window is ready


    for (int k = 0; k < 20; k++) {

        bool local_flag = true;  // Each process sets this based on its logic
        bool global_flag = true;
        set<int> A = buckets[k];

        // Process bucket
        while(global_flag){
            set<int> A_prim;
            for (int u: A) {
                Vertex &current_vertex = vertex_mapping[u];

                long long d_u;
                d_u = local_d[local_index(u, num_procs)];

                for (Edge e : current_vertex.edges) {
                    // Relax edge
                    int v = e.v2;
                    long long w = e.weight;
                    
                    long long d_v;
                    // Read current d[v]
                    MPI_Win_lock(MPI_LOCK_SHARED, owner(v, num_procs), 0, win_d);
                    MPI_Get(&d_v, 1, MPI_LONG_LONG, owner(v, num_procs),
                            local_index(v, num_procs), 1, MPI_LONG_LONG, win_d);
                    MPI_Win_unlock(owner(v, num_procs), win_d);


                    int old_bucket = d_v / delta;
                    long long old_d = d_v;
                    long long new_d = min(d_v, d_u + w);
                    cout << "my rank: " << rank << ",new min updated: " << new_d << endl;

                    int updated = 1;

                    if (new_d < old_d) {
                        // Update remote d_v
                        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, owner(v, num_procs), 0, win_d);
                        MPI_Put(&new_d, 1, MPI_LONG_LONG, owner(v, num_procs),
                                local_index(v, num_procs), 1, MPI_LONG_LONG, win_d);
                        MPI_Win_unlock(owner(v, num_procs), win_d);

                        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, owner(v, num_procs), 0, win_changed);
                        MPI_Put(&updated, 1, MPI_INT, owner(v, num_procs),
                                local_index(v, num_procs), 1, MPI_INT, win_changed);
                        MPI_Win_unlock(owner(v, num_procs), win_changed);

                        local_flag = true;

                    }
                }
            }


            MPI_Barrier(MPI_COMM_WORLD);

            if (!local_flag) {

                for (int i = 0; i < local_vertex_count; i++) {
                    if (local_changed[i] == 1) {
                            long long old_bucket = local_d_prev[i] / delta;
                            long long new_bucket = local_d[i] / delta

                        if (new_bucket < old_bucket) {
                            buckets[old_bucket].erase(i);
                            buckets[new_bucket].insert(i);
                            A_prim.insert(i);
                            local_d_prev[i] = local_d[i];
                            local_changed[i] = 0;
                        }
                    }
                }
            }

            A.clear();

            set_intersection(A_prim.begin(), A_prim.end(),
                          buckets[k].begin(), buckets[k].end(),
                          inserter(A, A.begin()));

            if (A.empty()) {
                local_flag = false;
            }
            
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Allreduce(&local_flag, &global_flag, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
        }

        buckets[k].clear();
    }
    MPI_Win_free(&win_d);
    MPI_Win_free(&win_changed);

    unordered_map<int, long long> result;
    for (int i = 0; i < local_vertex_count; ++i) {
        int global_id = i * num_procs + rank;
        result[global_id] = local_d[i];
    }

    return result;
}


// unordered_map<int, long long> delta_stepping(
//     unordered_map<int, Vertex> vertex_mapping,
//     int root, int rank, int num_procs, int local_vertex_count
// ) {
//     vector<long long> local_d(local_vertex_count, INF);

//     MPI_Win win;
//     MPI_Win_create(local_d.data(), local_vertex_count * sizeof(long long),
//                    sizeof(long long), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

//     // Use lock_all/unlock_all for simpler sync
//     MPI_Win_lock_all(0, win);

//     unordered_map<long long, set<int>> buckets;

//     // Initialize root
//     if (owner(root, num_procs) == rank) {
//         local_d[local_index(root, num_procs)] = 0;
//         buckets[0].insert(root);
//     }

//     const int max_buckets = 1000;
//     int global_active = 1;

//     for (int k = 0; k < max_buckets; ++k) {
//         int local_active = (buckets.count(k) && !buckets[k].empty()) ? 1 : 0;

//         // Termination check
//         MPI_Allreduce(&local_active, &global_active, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
//         if (!global_active) break;

//         set<int> A = buckets[k];
//         buckets[k].clear();

//         while (!A.empty()) {
//             set<int> A_next;

//             for (int u : A) {
//                 if (vertex_mapping.find(u) == vertex_mapping.end()) continue;

//                 Vertex& current_vertex = vertex_mapping[u];
//                 long long d_u = local_d[local_index(u, num_procs)];

//                 for (const Edge& e : current_vertex.edges) {
//                     int v = e.v2;
//                     long long w = e.weight;

//                     long long d_v;

//                     // Read remote d[v]
//                     MPI_Get(&d_v, 1, MPI_LONG_LONG,
//                             owner(v, num_procs),
//                             local_index(v, num_procs),
//                             1, MPI_LONG_LONG, win);
//                     MPI_Win_flush(owner(v, num_procs), win);

//                     long long new_d = d_u + w;

//                     if (new_d < d_v) {
//                         // Write new value
//                         MPI_Put(&new_d, 1, MPI_LONG_LONG,
//                                 owner(v, num_procs),
//                                 local_index(v, num_procs),
//                                 1, MPI_LONG_LONG, win);
//                         MPI_Win_flush(owner(v, num_procs), win);

//                         // Locally track updates
//                         if (owner(v, num_procs) == rank) {
//                             local_d[local_index(v, num_procs)] = new_d;
//                             buckets[new_d / delta].insert(v);
//                             A_next.insert(v);
//                         }
//                     }
//                 }
//             }

//             A = A_next;
//         }
//     }

//     MPI_Win_unlock_all(win);
//     MPI_Win_free(&win);

//     unordered_map<int, long long> result;
//     for (int i = 0; i < local_vertex_count; ++i) {
//         int global_id = i * num_procs + rank;
//         result[global_id] = local_d[i];
//     }

//     return result;
// }


// unordered_map<int, long long> delta_stepping(unordered_map<int, Vertex> vertex_mapping, int root, int rank, int num_procs, int local_vertex_count) {    
//     vector<long long> local_d(local_vertex_count, INF);
//     vector<long long> local_changed(local_vertex_count, 0);

//     // Setup MPI Window for local_d
//     MPI_Win win;
//     MPI_Win_create(local_d.data(), local_vertex_count * sizeof(long long),
//                    sizeof(long long), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

//     MPI_Win_create(local_changed.data(), local_vertex_count * sizeof(int),
//                     sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);



//     set<int> zero_set;
//     set<int> inf_set;

//     unordered_map<long long, set<int>> buckets;

//     if (owner(root, num_procs) == rank) {
//         local_d[local_index(root, num_procs)] = 0;
//         buckets[0].insert(root);
//     }

//     cout << "MY RANK IS: " << rank << "\n";

//     MPI_Barrier(MPI_COMM_WORLD); // Ensure window is ready




//     for (int k = 0; k < 20; k++) {

//         if (buckets.count(k) == 0) {
//             cout << "BUCKET SKIPPED " << k << "; my rank is: " << rank << "\n";
//             continue;
//         }

//         set<int> A = buckets[k];

//         // Process bucket
//         while(!A.empty()){
//             set<int> A_prim;
//             for (int u: A) {
//                 Vertex &current_vertex = vertex_mapping[u];

//                 long long d_u;
//                 d_u = local_d[local_index(u, num_procs)];

//                 for (Edge e : current_vertex.edges) {
//                     // Relax edge
//                     int v = e.v2;
//                     long long w = e.weight;
                    
//                     long long d_v;
//                     // Read current d[v]
//                     MPI_Win_lock(MPI_LOCK_SHARED, owner(v, num_procs), 0, win);
//                     MPI_Get(&d_v, 1, MPI_LONG_LONG, owner(v, num_procs),
//                             local_index(v, num_procs), 1, MPI_LONG_LONG, win);
//                     MPI_Win_unlock(owner(v, num_procs), win);


//                     int old_bucket = d_v / delta;
//                     long long old_d = d_v;
//                     long long new_d = min(d_v, d_u + w);
//                     cout << "my rank: " << rank << ",new min updated: " << new_d << endl;

//                     if (new_d < old_d) {
//                         // Update remote d_v
//                         MPI_Win_lock(MPI_LOCK_EXCLUSIVE, owner(v, num_procs), 0, win);
//                         MPI_Put(&new_d, 1, MPI_LONG_LONG, owner(v, num_procs),
//                                 local_index(v, num_procs), 1, MPI_LONG_LONG, win);
//                         MPI_Win_unlock(owner(v, num_procs), win);

//                         if (owner(v, num_procs) == rank) {
//                             long long old_bucket = d_v / delta;
//                             long long new_bucket = new_d / delta;

//                             if (new_bucket < old_bucket) {
//                                 buckets[old_bucket].erase(v);
//                                 buckets[new_bucket].insert(v);
//                                 A_prim.insert(v);
//                             }
//                         }

//                     }
//                 }
//             }

//             A.clear();

//             set_intersection(A_prim.begin(), A_prim.end(),
//                           buckets[k].begin(), buckets[k].end(),
//                           inserter(A, A.begin()));

            
//         }

//         buckets[k].clear();
//     }
//     MPI_Win_free(&win);

//     unordered_map<int, long long> result;
//     for (int i = 0; i < local_vertex_count; ++i) {
//         int global_id = i * num_procs + rank;
//         result[global_id] = local_d[i];
//     }

//     return result;
// }


// unordered_map_seq<int, long long> delta_stepping(unordered_map<int, Vertex> vertex_mapping, int root, int rank) {
//     unordered_map<int, long long> d;
//     unordered_map<long long, set<int>> buckets;


//     set<int> zero_set;
//     set<int> inf_set;

//     buckets[INF] = zero_set;
//     for (auto& it: vertex_mapping) {
//         if (root == root) {
//             d[v.id] = 0;
//             buckets[0] = zero_set;
//             buckets[0].insert(it.first);
//         }
//         else {
//             d[it.first] = INF;
//         }
//     }


//     for (int k = 0; k < INF; k++) {

//         if (buckets.count(k) == 0) {
//             continue;
//         }

//         set<int> A = buckets[k];

//         // Process bucket
//         while(!A.empty()){
//             set<int> A_prim;
//             for (int u: A) {
//                 Vertex current_vertex = vertex_mapping[u];
//                 vector<Edge> edges = current_vertex.edges;

//                 for (Edge e: edges) {
//                     // Relax edge
//                     int u = e.v1;
//                     int v = e.v2;
//                     long long w = e.weight;

//                     int old_bucket = d[v]/delta;
//                     long long old_d = d[v];
//                     d[v] = min(d[v], d[u] + w);

//                     if (d[u] + w < d[v]) {
//                         long long old_bucket = d[v] / delta;
//                         d[v] = d[u] + w;
//                         long long new_bucket = d[v] / delta;

//                         if(new_bucket < old_bucket) {
//                             buckets[old_bucket].erase(v);

//                             if (buckets.count(new_bucket) == 0) {
//                                 set<int> new_set;
//                                 buckets[new_bucket] = new_set;
//                             }
//                             buckets[new_bucket].insert(v);
//                         }

//                         A_prim.insert(v);

//                     }
                    
//                 }
//             }

//             A.clear();

//             set_intersection(A_prim.begin(), A_prim.end(),
//                           buckets[k].begin(), buckets[k].end(),
//                           inserter(A, A.begin()));
//         }

//         buckets[k].clear();
//     }

//     return d;
// }






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

    cout << "My rank is: " << rank << "\n";

    std::ifstream infile(input_file);
    if (!infile.is_open()) {
        std::cerr << "Rank " << rank << ": Failed to open input file " << input_file << "\n";
        MPI_Finalize();
        return 1;
    }

    int num_vertices, start_vertex, end_vertex;
    infile >> num_vertices >> start_vertex >> end_vertex;

    unordered_map<int, Vertex> my_vertices;

    for (int i = start_vertex; i <= end_vertex; i++) {
        Vertex v;
        v.id = i;
        vector<Edge> edges;
        v.edges = edges;
        my_vertices[i] = v;
    }

    int u;
    int v;
    long long w;

    int local_vertex_count = end_vertex - start_vertex + 1;
    int num_procs = num_vertices/local_vertex_count;

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
            my_vertices[v].edges.push_back(e);
        }    

    }
    infile.close();

    unordered_map<int, long long> final_values = delta_stepping(my_vertices, global_root, rank, num_procs, local_vertex_count);

    // Dummy output for testing (write -1 as shortest path for each vertex)
    std::ofstream outfile(output_file);
    if (!outfile.is_open()) {
        std::cerr << "Rank " << rank << ": Failed to open output file " << output_file << "\n";
        MPI_Finalize();
        return 1;
    }

    outfile << "shit 5" << "\n";

    // for (int v = start_vertex; v <= end_vertex; ++v) {
    //     outfile << v << " edges:";
    //     Vertex vertex = my_vertices[v];
    //     for (Edge e: vertex.edges) {
    //         outfile << e.v2 << " w: " << e.weight << ";";
    //     }
    //     outfile << "\n";
    // }
    for (int v = start_vertex; v <= end_vertex; ++v) {
        outfile << v << " " << final_values[v] << "\n";
    }
    outfile.close();

    MPI_Finalize();
    return 0;
}
