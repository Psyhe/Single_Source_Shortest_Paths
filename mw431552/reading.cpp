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

const long long INF = 1e18;
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

int owner(int vertex, int total_nodes, int num_procs) {
    int base = total_nodes / num_procs;
    int remainder = total_nodes % num_procs;

    // The first 'remainder' ranks have (base + 1) vertices
    int threshold = (base + 1) * remainder;

    if (vertex < threshold) {
        return vertex / (base + 1);
    } else {
        return remainder + (vertex - threshold) / base;
    }
}

int vertices_for_rank(int rank, int total_nodes, int num_procs) {
    int base = total_nodes / num_procs;
    int remainder = total_nodes % num_procs;

    // Ranks [0, remainder - 1] get one extra vertex
    if (rank < remainder) {
        return base + 1;
    } else {
        return base;
    }
}

int global_to_local_index(int global_v, int rank, int total_vertices, int num_procs) {
    int base = total_vertices / num_procs;
    int remainder = total_vertices % num_procs;

    int offset;
    if (rank < remainder) {
        offset = rank * (base + 1);
    } else {
        offset = remainder * (base + 1) + (rank - remainder) * base;
    }

    return global_v - offset;
}

int local_to_global_index(int local_idx, int rank, int total_vertices, int num_procs) {
    int base = total_vertices / num_procs;
    int remainder = total_vertices % num_procs;

    int offset;
    if (rank < remainder) {
        offset = rank * (base + 1);
    } else {
        offset = remainder * (base + 1) + (rank - remainder) * base;
    }

    return offset + local_idx;
}


int local_index(int v, int local_vertex_count) {
    return v % local_vertex_count;
}

void relax_edge(
    int u, Edge& e, int rank, int num_vertices, int num_procs,
    unordered_map<int, Vertex>& vertex_mapping,
    vector<long long>& local_d, vector<long long>& local_changed,
    vector<long long>& local_d_prev,
    MPI_Win& win_d, MPI_Win& win_changed
) {
    int local_vertex_count = local_d.size();
    long long d_u = local_d[local_index(u, local_vertex_count)];

    int v = e.v2;
    long long w = e.weight;

    int owner_rank = owner(v, num_vertices, num_procs);
    int local_idx = global_to_local_index(v, owner_rank, num_vertices, num_procs);

    long long d_v;
    MPI_Win_lock(MPI_LOCK_SHARED, owner_rank, 0, win_d);
    MPI_Get(&d_v, 1, MPI_LONG_LONG, owner_rank, local_idx, 1, MPI_LONG_LONG, win_d);
    MPI_Win_unlock(owner_rank, win_d);

    long long new_d = min(d_v, d_u + w);

    if (new_d < d_v) {
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, owner_rank, 0, win_d);
        MPI_Put(&new_d, 1, MPI_LONG_LONG, owner_rank, local_idx, 1, MPI_LONG_LONG, win_d);
        MPI_Win_flush(owner_rank, win_d);
        MPI_Win_unlock(owner_rank, win_d);

        long long updated = 1;
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, owner_rank, 0, win_changed);
        MPI_Put(&updated, 1, MPI_LONG_LONG, owner_rank, local_idx, 1, MPI_LONG_LONG, win_changed);
        MPI_Win_flush(owner_rank, win_changed);
        MPI_Win_unlock(owner_rank, win_changed);

        if (owner_rank == rank) {
            local_d[local_idx] = new_d;
            local_changed[local_idx] = 1;
        }
    }
}

void process_bucket(
    const set<int>& A, unordered_map<int, Vertex>& vertex_mapping,
    int rank, int num_vertices, int num_procs,
    vector<long long>& local_d, vector<long long>& local_changed,
    vector<long long>& local_d_prev,
    MPI_Win& win_d, MPI_Win& win_changed
) {
    for (int u : A) {
        Vertex& current_vertex = vertex_mapping[u];
        for (Edge& e : current_vertex.edges) {
            relax_edge(u, e, rank, num_vertices, num_procs,
                       vertex_mapping, local_d, local_changed, local_d_prev,
                       win_d, win_changed);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

void relax_edge_IOS(
    int u, Edge& e, int rank, int num_vertices, int num_procs,
    unordered_map<int, Vertex>& vertex_mapping,
    vector<long long>& local_d, vector<long long>& local_changed,
    vector<long long>& local_d_prev,
    MPI_Win& win_d, MPI_Win& win_changed, long long k
) {
    int local_vertex_count = local_d.size();
    long long d_u = local_d[local_index(u, local_vertex_count)];

    int v = e.v2;
    long long w = e.weight;

    int owner_rank = owner(v, num_vertices, num_procs);
    int local_idx = global_to_local_index(v, owner_rank, num_vertices, num_procs);

    long long d_v;
    MPI_Win_lock(MPI_LOCK_SHARED, owner_rank, 0, win_d);
    MPI_Get(&d_v, 1, MPI_LONG_LONG, owner_rank, local_idx, 1, MPI_LONG_LONG, win_d);
    MPI_Win_unlock(owner_rank, win_d);

    long long new_d = min(d_v, d_u + w);

    if ((new_d < d_v) && (new_d <= ((k+1) * delta - 1))) {
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, owner_rank, 0, win_d);
        MPI_Put(&new_d, 1, MPI_LONG_LONG, owner_rank, local_idx, 1, MPI_LONG_LONG, win_d);
        MPI_Win_flush(owner_rank, win_d);
        MPI_Win_unlock(owner_rank, win_d);

        long long updated = 1;
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, owner_rank, 0, win_changed);
        MPI_Put(&updated, 1, MPI_LONG_LONG, owner_rank, local_idx, 1, MPI_LONG_LONG, win_changed);
        MPI_Win_flush(owner_rank, win_changed);
        MPI_Win_unlock(owner_rank, win_changed);

        if (owner_rank == rank) {
            local_d[local_idx] = new_d;
            local_changed[local_idx] = 1;
        }
    }
}

void process_bucket_first_phase_IOS(
    const set<int>& A, unordered_map<int, Vertex>& vertex_mapping,
    int rank, int num_vertices, int num_procs,
    vector<long long>& local_d, vector<long long>& local_changed,
    vector<long long>& local_d_prev,
    MPI_Win& win_d, MPI_Win& win_changed, long long k
) {
    for (int u : A) {
        Vertex& current_vertex = vertex_mapping[u];
        for (Edge& e : current_vertex.edges) {
            if (e.weight < delta) {
                relax_edge_IOS(u, e, rank, num_vertices, num_procs,
                        vertex_mapping, local_d, local_changed, local_d_prev,
                        win_d, win_changed, k);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

set<int> update_buckets_and_collect_active_set(
    vector<long long>& local_d, vector<long long>& local_changed,
    vector<long long>& local_d_prev, unordered_map<long long, set<int>>& buckets,
    int rank, int num_vertices, int num_procs, long long k
) {
    set<int> A_prim;
    int local_vertex_count = local_d.size();

    for (int i = 0; i < local_vertex_count; i++) {
        if (local_changed[i] == 1) {
            long long old_bucket = local_d_prev[i] / delta;
            long long new_bucket = local_d[i] / delta;
            int global_id = local_to_global_index(i, rank, num_vertices, num_procs);

            if (new_bucket < old_bucket) {
                buckets[old_bucket].erase(global_id);
                buckets[new_bucket].insert(global_id);
            }

            if (local_d_prev[i] > local_d[i]) {
                A_prim.insert(global_id);
            }

            local_d_prev[i] = local_d[i];
            local_changed[i] = 0;
        }
    }

    return A_prim;
}


unordered_map<int, long long> delta_stepping_basic(unordered_map<int, Vertex> vertex_mapping, int root, int rank, int num_procs, int num_vertices) {
    int local_vertex_count = vertices_for_rank(rank, num_vertices, num_procs);
    vector<long long> local_d(local_vertex_count, INF * delta);
    vector<long long> local_changed(local_vertex_count, 0);
    vector<long long> local_d_prev(local_vertex_count, INF * delta);

    // Setup MPI Windows
    MPI_Win win_d, win_changed;

    MPI_Win_create(local_d.data(), local_vertex_count * sizeof(long long),
                   sizeof(long long), MPI_INFO_NULL, MPI_COMM_WORLD, &win_d);

    MPI_Win_create(local_changed.data(), local_vertex_count * sizeof(long long),
                   sizeof(long long), MPI_INFO_NULL, MPI_COMM_WORLD, &win_changed);

    unordered_map<long long, set<int>> buckets;

    int starting_point = rank * local_vertex_count;
    for (int i = 0; i < local_vertex_count; ++i) {
        int global_id = starting_point + i;
        if (global_id != root) {
            buckets[INF].insert(global_id);
        }
    }

    if (owner(root, num_vertices, num_procs) == rank) {
        int li = local_index(root, local_vertex_count);
        local_d[li] = 0;
        local_d_prev[li] = 0;
        buckets[0].insert(root);
    }

    MPI_Barrier(MPI_COMM_WORLD); // Ensure window is ready

    long long k = 0;
    bool continue_running = true;

    while (continue_running) {
        set<int> A = buckets[k];

        bool filled_buckets = 0;
        for (const auto& b : buckets) {
            filled_buckets |= (!b.second.empty());
        }

        MPI_Allreduce(&filled_buckets, &continue_running, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
        if (!continue_running) break;

        bool local_flag = true, global_flag = true;

        while (global_flag) {
            process_bucket(A, vertex_mapping, rank, num_vertices, num_procs,
                        local_d, local_changed, local_d_prev, win_d, win_changed);

            set<int> A_prim = update_buckets_and_collect_active_set(
                local_d, local_changed, local_d_prev, buckets,
                rank, num_vertices, num_procs, k
            );

            A.clear();
            set_intersection(A_prim.begin(), A_prim.end(), buckets[k].begin(), buckets[k].end(),
                            inserter(A, A.begin()));

            local_flag = !A.empty();
            MPI_Allreduce(&local_flag, &global_flag, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
        }

        buckets[k].clear();
        k += 1;
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Win_free(&win_d);
    MPI_Win_free(&win_changed);

    unordered_map<int, long long> result;
    starting_point = rank * local_vertex_count;
    for (int i = 0; i < local_vertex_count; ++i) {
        int global_id = starting_point + i;
        result[global_id] = local_d[i];
    }

    return result;
}

unordered_map<int, long long> delta_stepping_IOS(unordered_map<int, Vertex> vertex_mapping, int root, int rank, int num_procs, int num_vertices) {
    int local_vertex_count = vertices_for_rank(rank, num_vertices, num_procs);
    vector<long long> local_d(local_vertex_count, INF * delta);
    vector<long long> local_changed(local_vertex_count, 0);
    vector<long long> local_d_prev(local_vertex_count, INF * delta);

    // Setup MPI Windows
    MPI_Win win_d, win_changed;

    MPI_Win_create(local_d.data(), local_vertex_count * sizeof(long long),
                   sizeof(long long), MPI_INFO_NULL, MPI_COMM_WORLD, &win_d);

    MPI_Win_create(local_changed.data(), local_vertex_count * sizeof(long long),
                   sizeof(long long), MPI_INFO_NULL, MPI_COMM_WORLD, &win_changed);

    unordered_map<long long, set<int>> buckets;

    int starting_point = rank * local_vertex_count;
    for (int i = 0; i < local_vertex_count; ++i) {
        int global_id = starting_point + i;
        if (global_id != root) {
            buckets[INF].insert(global_id);
        }
    }

    if (owner(root, num_vertices, num_procs) == rank) {
        int li = local_index(root, local_vertex_count);
        local_d[li] = 0;
        local_d_prev[li] = 0;
        buckets[0].insert(root);
    }

    MPI_Barrier(MPI_COMM_WORLD); // Ensure window is ready

    long long k = 0;
    bool continue_running = true;

    while (continue_running) {
        set<int> A = buckets[k];

        bool filled_buckets = 0;
        for (const auto& b : buckets) {
            filled_buckets |= (!b.second.empty());
        }

        MPI_Allreduce(&filled_buckets, &continue_running, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
        if (!continue_running) break;

        bool local_flag = true, global_flag = true;

        while (global_flag) {
            process_bucket_first_phase_IOS(A, vertex_mapping, rank, num_vertices, num_procs,
                        local_d, local_changed, local_d_prev, win_d, win_changed, k);

            set<int> A_prim = update_buckets_and_collect_active_set(
                local_d, local_changed, local_d_prev, buckets,
                rank, num_vertices, num_procs, k
            );

            A.clear();
            set_intersection(A_prim.begin(), A_prim.end(), buckets[k].begin(), buckets[k].end(),
                            inserter(A, A.begin()));

            local_flag = !A.empty();
            MPI_Allreduce(&local_flag, &global_flag, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
        }

        process_bucket(A, vertex_mapping, rank, num_vertices, num_procs,
                        local_d, local_changed, local_d_prev, win_d, win_changed);
        
        set<int> A_prim = update_buckets_and_collect_active_set(
            local_d, local_changed, local_d_prev, buckets,
            rank, num_vertices, num_procs, k
        );

        buckets[k].clear();
        k += 1;
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Win_free(&win_d);
    MPI_Win_free(&win_changed);

    unordered_map<int, long long> result;
    starting_point = rank * local_vertex_count;
    for (int i = 0; i < local_vertex_count; ++i) {
        int global_id = starting_point + i;
        result[global_id] = local_d[i];
    }

    return result;
}



int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int num_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

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

    cout << "Processing with IOS" << endl;
    unordered_map<int, long long> final_values = delta_stepping_IOS(my_vertices, global_root, rank, num_processes, num_vertices);

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
