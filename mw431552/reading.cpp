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
#include <numeric>


using namespace std;

const long long INF = 1e18;
int global_root = 0;
const double tau = 0.4;

int delta = 5; // TODO fine-tune

struct Edge {
    int v1, v2;
    long long weight;
};

struct Vertex {
    int id;
    vector<Edge> edges;
    int degree;
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


// int local_index(int v, int local_vertex_count) {
//     return v % local_vertex_count;
// }

void relax_edge(
    int u, Edge& e, int rank, int num_vertices, int num_procs,
    unordered_map<int, Vertex>& vertex_mapping,
    vector<long long>& local_d, vector<long long>& local_changed,
    vector<long long>& local_d_prev,
    MPI_Win& win_d, MPI_Win& win_changed
) {
    int local_vertex_count = local_d.size();
    long long d_u = local_d[global_to_local_index(u, rank, num_vertices, num_procs)];

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
        // Take exclusive lock, re-check condition, then write
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, owner_rank, 0, win_d);
        long long current_dv;
        MPI_Get(&current_dv, 1, MPI_LONG_LONG, owner_rank, local_idx, 1, MPI_LONG_LONG, win_d);
        MPI_Win_flush(owner_rank, win_d);

        if (new_d < current_dv) {
            MPI_Put(&new_d, 1, MPI_LONG_LONG, owner_rank, local_idx, 1, MPI_LONG_LONG, win_d);
            MPI_Win_flush(owner_rank, win_d);
        }
        MPI_Win_unlock(owner_rank, win_d);

        if (new_d < current_dv) {
            long long updated = 1;
            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, owner_rank, 0, win_changed);
            MPI_Put(&updated, 1, MPI_LONG_LONG, owner_rank, local_idx, 1, MPI_LONG_LONG, win_changed);
            MPI_Win_flush(owner_rank, win_changed);
            MPI_Win_unlock(owner_rank, win_changed);

            // if (owner_rank == rank) {
            //     local_d[local_idx] = new_d;
            //     local_changed[local_idx] = 1;
            // }
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

set<int> update_set_and_collect_active(
    vector<long long>& local_d, vector<long long>& local_changed,
    vector<long long>& local_d_prev, set<int>& current_set,
    int rank, int num_vertices, int num_procs
) {
    set<int> A_prim;
    int local_vertex_count = local_d.size();

    for (int i = 0; i < local_vertex_count; i++) {
        if (local_changed[i] == 1) {
            int global_id = local_to_global_index(i, rank, num_vertices, num_procs);
            // cout << "inside update rank:" << rank << " local_id = "<< i << endl;
            // cout << "inside update rank:" << rank << " global_id = "<< global_id << endl;
            // cout << "local_d_prev[i] " << local_d_prev[i] << endl;
            // cout << "local_d[i] " << local_d[i] << endl;


            if (local_d_prev[i] > local_d[i]) {
                current_set.insert(global_id);
            }

            if (local_d_prev[i] > local_d[i]) {
                A_prim.insert(global_id);
            }

            local_d_prev[i] = local_d[i];
            local_changed[i] = 0;

            // cout << "Inside update current set" << endl;
            for (int it: current_set) {
                // cout << it << " ";
            }
            // cout << endl;
        }
    }

    return A_prim;
}


unordered_map<int, long long> delta_stepping_basic(unordered_map<int, Vertex> vertex_mapping, int root, int rank, int num_procs, int num_vertices) {
    int local_vertex_count = vertices_for_rank(rank, num_vertices, num_procs);
    vector<long long> local_d(local_vertex_count, INF);
    vector<long long> local_changed(local_vertex_count, 0);
    vector<long long> local_d_prev(local_vertex_count, INF);

    // Setup MPI Windows
    MPI_Win win_d, win_changed;

    MPI_Win_create(local_d.data(), local_vertex_count * sizeof(long long),
                   sizeof(long long), MPI_INFO_NULL, MPI_COMM_WORLD, &win_d);

    MPI_Win_create(local_changed.data(), local_vertex_count * sizeof(long long),
                   sizeof(long long), MPI_INFO_NULL, MPI_COMM_WORLD, &win_changed);

    unordered_map<long long, set<int>> buckets;

    int starting_point = local_to_global_index(0, rank, num_vertices, num_procs);
    for (int i = 0; i < local_vertex_count; ++i) {
        int global_id = starting_point + i;
        if (global_id != root) {
            buckets[INF/delta].insert(global_id);
        }
    }

    if (owner(root, num_vertices, num_procs) == rank) {
        int li = global_to_local_index(root, rank, num_vertices, num_procs);
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

        int local_flag = 1, global_flag = 1;

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

            // // cout << "A_prim rank: " << rank << " set size: " << A_prim.size()  << endl;
            // for (auto &it: A_prim) {
            //     // cout << it << " ";
            // }
            // // cout << endl;


            // // cout << "buckets rank: " << rank << " set size: " << buckets[k].size()  <<endl;
            // for (auto &it: buckets[k]) {
            //     // cout << it << " ";
            // }
            // // cout << endl;

            // // cout << "A rank: " << rank << " set size: " << A.size()  <<endl;
            // for (auto &it: A) {
            //     // cout << it << " ";
            // }
            // // cout << endl;

            // cout << "Buckets in rank: " << rank << endl;
            for (auto &it: buckets) {
                if (!(buckets[it.first]).empty() && it.first==INF/delta) {
                    // cout << "k" << it.first << endl;
                    for (auto &it1: buckets[it.first]) {
                        // cout << it1 << " ";
                    }
                }
            }
            // cout << endl;

            local_flag = !(A.size() == 0);
            // cout << "Local flag: " << local_flag << "rank: " << rank << " set size: " << A.size()  <<endl;
            cout<< "Current k: " << k << endl;
            MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        }

        buckets[k].clear();
        k += 1;
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Win_free(&win_d);
    MPI_Win_free(&win_changed);

    // // cout << "rank: " << rank << endl;
    // for (auto &it: buckets) {
    //     // cout << "k" << it.first << endl;
    //     for (auto &it1: buckets[it.first]) {
    //         // cout << it1 << " ";
    //     }
    // }
    // // cout << endl;

    unordered_map<int, long long> result;
    starting_point = local_to_global_index(0, rank, num_vertices, num_procs);
    for (int i = 0; i < local_vertex_count; ++i) {
        int global_id = starting_point + i;
        result[global_id] = local_d[i];
    }

    return result;
}

void relax_edge_IOS(
    int u, Edge& e, int rank, int num_vertices, int num_procs,
    unordered_map<int, Vertex>& vertex_mapping,
    vector<long long>& local_d, vector<long long>& local_changed,
    vector<long long>& local_d_prev,
    MPI_Win& win_d, MPI_Win& win_changed, long long k
) {
    // cout << "relax IOS" << endl;
    int local_vertex_count = local_d.size();
    long long d_u = local_d[global_to_local_index(u, rank, num_vertices, num_procs)];

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
        // Take exclusive lock, re-check condition, then write
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, owner_rank, 0, win_d);
        long long current_dv;
        MPI_Get(&current_dv, 1, MPI_LONG_LONG, owner_rank, local_idx, 1, MPI_LONG_LONG, win_d);
        MPI_Win_flush(owner_rank, win_d);

        if (new_d < current_dv) {
            MPI_Put(&new_d, 1, MPI_LONG_LONG, owner_rank, local_idx, 1, MPI_LONG_LONG, win_d);
            MPI_Win_flush(owner_rank, win_d);
        }
        MPI_Win_unlock(owner_rank, win_d);

        if (new_d < current_dv) {
            long long updated = 1;
            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, owner_rank, 0, win_changed);
            MPI_Put(&updated, 1, MPI_LONG_LONG, owner_rank, local_idx, 1, MPI_LONG_LONG, win_changed);
            MPI_Win_flush(owner_rank, win_changed);
            MPI_Win_unlock(owner_rank, win_changed);

            // if (owner_rank == rank) {
            //     local_d[local_idx] = new_d;
            //     local_changed[local_idx] = 1;
            // }
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
            if (e.weight <= delta) {
                relax_edge_IOS(u, e, rank, num_vertices, num_procs,
                        vertex_mapping, local_d, local_changed, local_d_prev,
                        win_d, win_changed, k);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}


unordered_map<int, long long> delta_stepping_IOS(unordered_map<int, Vertex> vertex_mapping, int root, int rank, int num_procs, int num_vertices) {
    int local_vertex_count = vertices_for_rank(rank, num_vertices, num_procs);
    vector<long long> local_d(local_vertex_count, INF);
    vector<long long> local_changed(local_vertex_count, 0);
    vector<long long> local_d_prev(local_vertex_count);

    // Setup MPI Windows
    MPI_Win win_d, win_changed;

    MPI_Win_create(local_d.data(), local_vertex_count * sizeof(long long),
                   sizeof(long long), MPI_INFO_NULL, MPI_COMM_WORLD, &win_d);

    MPI_Win_create(local_changed.data(), local_vertex_count * sizeof(long long),
                   sizeof(long long), MPI_INFO_NULL, MPI_COMM_WORLD, &win_changed);

    unordered_map<long long, set<int>> buckets;

    int starting_point = local_to_global_index(0, rank, num_vertices, num_procs);

    for (int i = 0; i < local_vertex_count; ++i) {
        int global_id = starting_point + i;
        if (global_id != root) {
            buckets[INF/delta].insert(global_id);
        }
    }

    if (owner(root, num_vertices, num_procs) == rank) {
        int li = global_to_local_index(root, rank, num_vertices, num_procs);
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
        process_bucket(buckets[k], vertex_mapping, rank, num_vertices, num_procs,
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
    starting_point = local_to_global_index(0, rank, num_vertices, num_procs);

    for (int i = 0; i < local_vertex_count; ++i) {
        int global_id = starting_point + i;
        result[global_id] = local_d[i];
    }

    return result;
}


void process_bucket_pull_model(
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

long long local_push(
    const set<int>& current_bucket,
    const unordered_map<int, Vertex>& vertex_mapping,
    const vector<long long>& local_d,
    long long k,                // current bucket index
    double delta,
    int w_max                   // max edge weight
) {
    long long push_volume = 0;

    for (int u : current_bucket) {
        auto it = vertex_mapping.find(u);
        if (it != vertex_mapping.end()) {
            const Vertex& v = it->second;
            for (const Edge& e : v.edges) {
                if (e.weight > delta) {
                    push_volume++;
                }
            }
        }
    }

    return push_volume;
}

long long local_pull(
    const set<int>& current_bucket,
    const unordered_map<int, Vertex>& vertex_mapping,
    const vector<long long>& local_d,
    long long k,                // current bucket index
    double delta,
    int w_max,
    int rank,
    int num_vertices,
    int num_procs                  // max edge weight
) {
    long long pull_volume = 0;
    long long pull_requests = 0;

    for (auto& it: vertex_mapping) {
        int v_id = it.first;
        Vertex vertex = it.second;

        long long d_v = local_d[global_to_local_index(v_id, rank, num_vertices, num_procs)]; // local_d is local per process

        // Only consider vertices that are *not* in current bucket
        if ((d_v / delta) > k) {
            double range_upper = d_v - (k + 1) * delta;
            if (range_upper > 0) {
                double fraction = range_upper / w_max;
                pull_requests += static_cast<long long>(vertex.edges.size() * fraction);
            }
        }
    }

    pull_volume = 2 * pull_requests;

    return pull_volume;
}

// bool should_use_pull_model(
//     const set<int>& current_bucket,
//     const unordered_map<int, Vertex>& vertex_mapping,
//     const vector<long long>& local_d,
//     long long k,                // current bucket index
//     double delta,
//     int w_max                   // max edge weight
// ) {
//     // Estimate communication cost for push and pull
//     long long push_volume = 0;
//     long long pull_requests = 0;

//     // --- Push Volume Estimation: total long edges in current bucket ---
//     for (int u : current_bucket) {
//         auto it = vertex_mapping.find(u);
//         if (it != vertex_mapping.end()) {
//             const Vertex& v = it->second;
//             for (const Edge& e : v.edges) {
//                 if (e.weight > delta) {
//                     push_volume++;
//                 }
//             }
//         }
//     }

//     // --- Pull Volume Estimation: requests from later buckets ---
//     // We assume edge weights are uniformly distributed in [0, w_max]
//     // So, for each vertex v, expected edges in [Δ, d(v) - kΔ - 1] = degree × ((d(v) - (k+1)Δ) / w_max)

//     // for (const auto& [v_id, vertex] : vertex_mapping) {
//     for (auto& it: vertex_mapping) {
//         int v_id = it.first;
//         Vertex vertex = it.second;

//         long long d_v = local_d[local_index(v_id, local_d.size())]; // local_d is local per process

//         // Only consider vertices that are *not* in current bucket
//         if ((d_v / delta) > k) {
//             double range_upper = d_v - (k + 1) * delta;
//             if (range_upper > 0) {
//                 double fraction = range_upper / w_max;
//                 pull_requests += static_cast<long long>(vertex.edges.size() * fraction);
//             }
//         }
//     }

//     // Responses ≈ requests (as mentioned in the paper)
//     long long pull_volume = 2 * pull_requests;

//     // cout << 
//     // Compare estimated volumes
//     return pull_volume < push_volume;
// }


void process_bucket_outer_short(
    const set<int>& A, unordered_map<int, Vertex>& vertex_mapping,
    int rank, int num_vertices, int num_procs,
    vector<long long>& local_d, vector<long long>& local_changed,
    vector<long long>& local_d_prev,
    MPI_Win& win_d, MPI_Win& win_changed
) {
    for (int u : A) {
        Vertex& current_vertex = vertex_mapping[u];
        for (Edge& e : current_vertex.edges) {
            if (e.weight <=  delta) {
                relax_edge(u, e, rank, num_vertices, num_procs,
                        vertex_mapping, local_d, local_changed, local_d_prev,
                        win_d, win_changed);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}


struct PullRequest {
    int requester_v; // global id of v
    int u;           // global id of u
    // long long current_k;
};

struct PullResponse {
    int v;
    int u;
    long long d_u;
    long long weight;
};

void pull_model_process_long_edges(
    long long k,
    unordered_map<int, Vertex>& vertex_mapping,
    vector<long long>& local_d,
    vector<long long>& local_changed,
    int rank,
    int num_procs,
    int num_vertices,
    int delta
) {
    // ==================== Step 1: Build Pull Requests ====================
    // cout << "Step 1: Build Pull Requests: " << rank << endl;
    vector<vector<PullRequest>> requests_to_send(num_procs);

    for (const auto& it : vertex_mapping) {
        int id = it.first;
        Vertex vertex = it.second;

        long long d_v = local_d[global_to_local_index(vertex.id, rank, num_vertices, num_procs)];
        if ((d_v / delta) > k) {
            for (const auto& edge : vertex.edges) {
                int u = edge.v2;
                long long w = edge.weight;

                if (w < d_v - k * delta) {  // pruning condition
                    int owner_u = owner(u, num_vertices, num_procs);
                    requests_to_send[owner_u].push_back({vertex.id, u});
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD); // Ensure window is ready

    // ==================== Step 2: Exchange Pull Requests ====================
        // cout << "Step 2: Exchange Pull Requests: " << rank << endl;
        // cout<< "Current k: " << k << " for rank: " << rank << endl;

    vector<int> send_counts(num_procs), recv_counts(num_procs);
    vector<int> send_displs(num_procs), recv_displs(num_procs);

    for (int i = 0; i < num_procs; ++i)
    {
        for (auto xd : requests_to_send[i])
        // cout << "SENDING REQUEST TO PROC " << i << " FOR "  << " " << xd.requester_v << " " << xd.u << endl;
        send_counts[i] = requests_to_send[i].size();
    }

    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    int total_send = accumulate(send_counts.begin(), send_counts.end(), 0);
    int total_recv = accumulate(recv_counts.begin(), recv_counts.end(), 0);

    vector<PullRequest> flat_send_buf(total_send);
    vector<PullRequest> flat_recv_buf(total_recv);

    int offset = 0;
    for (int i = 0; i < num_procs; ++i) {
        send_displs[i] = offset;
        copy(requests_to_send[i].begin(), requests_to_send[i].end(), flat_send_buf.begin() + offset);
        offset += send_counts[i];
    }

    recv_displs[0] = 0;
    for (int i = 1; i < num_procs; ++i)
        recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];

    MPI_Alltoallv(
        flat_send_buf.data(), send_counts.data(), send_displs.data(), MPI_2INT,
        flat_recv_buf.data(), recv_counts.data(), recv_displs.data(), MPI_2INT,
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD); // Ensure window is ready

    // ==================== Step 3: Process Pull Requests ====================
    // cout << "Step 3: Process Pull Requests: " << rank << endl;

    vector<vector<PullResponse>> responses_to_send(num_procs);

    for (const auto& req : flat_recv_buf) {
        int v = req.requester_v;
        int u = req.u;

        if (vertex_mapping.count(u)) {
            long long d_u = local_d[global_to_local_index(u, rank, num_vertices, num_procs)];
            if ((d_u / delta) == k) {
                for (const Edge& e : vertex_mapping[u].edges) {
                    if (e.v2 == v) {
                        int owner_v = owner(v, num_vertices, num_procs);
                        // cout << "RESPONSE CREATED " << v << " " << u << " " << d_u << " " << e.weight << endl;
                        responses_to_send[owner_v].push_back({v, u, d_u, e.weight});
                        break;
                    }
                }
            }
        }
    }

    // cout << "Step 4: Exchange Pull Responses: " << rank << " current k: " << k << endl;
    MPI_Barrier(MPI_COMM_WORLD); // Ensure window is ready

    // ==================== Step 4: Exchange Pull Responses ====================
    for (int i = 0; i < num_procs; ++i)
        send_counts[i] = responses_to_send[i].size();

    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    total_send = accumulate(send_counts.begin(), send_counts.end(), 0);
    total_recv = accumulate(recv_counts.begin(), recv_counts.end(), 0);

    vector<PullResponse> flat_resp_send_buf(total_send);
    vector<PullResponse> flat_resp_recv_buf(total_recv);

    offset = 0;
    for (int i = 0; i < num_procs; ++i) {
        send_displs[i] = offset;
        copy(responses_to_send[i].begin(), responses_to_send[i].end(), flat_resp_send_buf.begin() + offset);
        offset += send_counts[i];
    }

    recv_displs[0] = 0;
    for (int i = 1; i < num_procs; ++i)
        recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];

    // Correct MPI datatype for PullResponse (int, long long, long long)
    MPI_Datatype MPI_PULL_RESP;
    int lengths[3] = {1, 1, 1};
    MPI_Aint displs[3];
    MPI_Datatype types[3] = {MPI_INT, MPI_LONG_LONG, MPI_LONG_LONG};

    PullResponse dummy;
    MPI_Aint base;
    MPI_Get_address(&dummy, &base);
    MPI_Get_address(&dummy.v, &displs[0]);
    MPI_Get_address(&dummy.d_u, &displs[1]);
    MPI_Get_address(&dummy.weight, &displs[2]);

    for (int i = 0; i < 3; ++i)
        displs[i] -= base;

    MPI_Type_create_struct(3, lengths, displs, types, &MPI_PULL_RESP);
    MPI_Type_commit(&MPI_PULL_RESP);

    MPI_Alltoallv(
        flat_resp_send_buf.data(), send_counts.data(), send_displs.data(), MPI_PULL_RESP,
        flat_resp_recv_buf.data(), recv_counts.data(), recv_displs.data(), MPI_PULL_RESP,
        MPI_COMM_WORLD
    );

    MPI_Type_free(&MPI_PULL_RESP);
    MPI_Barrier(MPI_COMM_WORLD); // Ensure window is ready

    // cout << " Step 5: Apply Responses: " << rank << " current k: " << k << endl;
    // ==================== Step 5: Apply Responses ====================
    for (const auto& resp : flat_resp_recv_buf) {
        // cout << "PROCESSING RESPONSE v:" << resp.v << " u:" << resp.u << " d_u:" << resp.d_u << " weight:" << resp.weight << endl;
        int v = resp.v;
        long long d_u = resp.d_u;
        long long w = resp.weight;

        int local_idx = global_to_local_index(v, rank, num_vertices, num_procs);
        long long& d_v = local_d[local_idx];
        long long new_d = d_u + w;
        // cout << "CURRENT INDEX" << v << endl;

        if (new_d < d_v) {
            local_d[local_idx] = new_d;
            local_changed[local_idx] = 1;
        }
    }
}


unordered_map<int, long long> delta_stepping_prunning(unordered_map<int, Vertex> vertex_mapping, int root, int rank, int num_procs, int num_vertices, long long local_max_weight) {
    // cout << "prunning" << endl;
    int local_vertex_count = vertices_for_rank(rank, num_vertices, num_procs);
    vector<long long> local_d(local_vertex_count, INF);
    vector<long long> local_changed(local_vertex_count, 0);
    vector<long long> local_d_prev(local_vertex_count, INF);

    long long real_max_weight;
    MPI_Allreduce(&local_max_weight, &real_max_weight, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

    // Setup MPI Windows
    MPI_Win win_d, win_changed;

    MPI_Win_create(local_d.data(), local_vertex_count * sizeof(long long),
                   sizeof(long long), MPI_INFO_NULL, MPI_COMM_WORLD, &win_d);

    MPI_Win_create(local_changed.data(), local_vertex_count * sizeof(long long),
                   sizeof(long long), MPI_INFO_NULL, MPI_COMM_WORLD, &win_changed);

    unordered_map<long long, set<int>> buckets;

    int starting_point = local_to_global_index(0, rank, num_vertices, num_procs);

    for (int i = 0; i < local_vertex_count; ++i) {
        int global_id = starting_point + i;
        if (global_id != root) {
            buckets[INF/delta].insert(global_id);
        }
    }

    if (owner(root, num_vertices, num_procs) == rank) {
        int li = global_to_local_index(root, rank, num_vertices, num_procs);
        local_d[li] = 0;
        local_d_prev[li] = 0;
        buckets[0].insert(root);
    }

    MPI_Barrier(MPI_COMM_WORLD); // Ensure window is ready

    long long k = 0;
    bool continue_running = true;

    while (continue_running && k < 200) {
        set<int> A = buckets[k];

        bool filled_buckets = 0;
        for (const auto& b : buckets) {
            filled_buckets |= (!b.second.empty());
        }

        MPI_Allreduce(&filled_buckets, &continue_running, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
        if (!continue_running) break;

        bool local_flag = true, global_flag = true;

        while (global_flag) {
            process_bucket_outer_short(A, vertex_mapping, rank, num_vertices, num_procs,
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

        long long local_push_count = local_push(buckets[k], vertex_mapping, local_d, k, delta, real_max_weight);
        long long local_pull_count = local_pull(buckets[k], vertex_mapping, local_d, k, delta, real_max_weight, rank, num_vertices, num_procs);

        long long total_push;
        long long total_pull;

        MPI_Barrier(MPI_COMM_WORLD); 
        MPI_Allreduce(&local_push_count, &total_push, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_pull_count, &total_pull, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);


        // if (true){
        if (total_pull < total_push) {
            //  outer short edge processing
            // process_bucket_outer_short(buckets[k], vertex_mapping, rank, num_vertices, num_procs,
            //     local_d, local_changed, local_d_prev, win_d, win_changed);
            
            pull_model_process_long_edges(k, vertex_mapping, local_d, local_changed, rank, num_procs, num_vertices, delta);
            
            MPI_Barrier(MPI_COMM_WORLD);

            set<int> A_prim = update_buckets_and_collect_active_set(
                local_d, local_changed, local_d_prev, buckets,
                rank, num_vertices, num_procs, k
            );

            // cout << "WHAT BUCKETS ARE LEFT:" << endl;
            for (auto &it: buckets) {
                // cout << "k: " << it.first << endl;
                for (auto &it1: buckets[it.first]) {
                    // cout << it1 << " ";
                }
                // cout << endl;
            }
        }
        else {
            // Default way
            process_bucket(buckets[k], vertex_mapping, rank, num_vertices, num_procs,
                local_d, local_changed, local_d_prev, win_d, win_changed);

            set<int> A_prim = update_buckets_and_collect_active_set(
                local_d, local_changed, local_d_prev, buckets,
                rank, num_vertices, num_procs, k
            );
        }

        buckets[k].clear();
        k += 1;
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Win_free(&win_d);
    MPI_Win_free(&win_changed);

    unordered_map<int, long long> result;
    starting_point = local_to_global_index(0, rank, num_vertices, num_procs);

    for (int i = 0; i < local_vertex_count; ++i) {
        int global_id = starting_point + i;
        result[global_id] = local_d[i];
    }

    return result;
}

long long update_weight(long long current_max, long long potential) {
    if (current_max >= potential) {
        return current_max;
    }
    else {
        return potential;
    }
}























unordered_map<int, long long> delta_stepping_hybrid(unordered_map<int, Vertex> vertex_mapping, int root, int rank, int num_procs, int num_vertices) {
    // cout << "hybrid" << endl;
    int local_vertex_count = vertices_for_rank(rank, num_vertices, num_procs);
    vector<long long> local_d(local_vertex_count, INF);
    vector<long long> local_changed(local_vertex_count, 0);
    vector<long long> local_d_prev(local_vertex_count, INF);

    // Setup MPI Windows
    MPI_Win win_d, win_changed;

    MPI_Win_create(local_d.data(), local_vertex_count * sizeof(long long),
                   sizeof(long long), MPI_INFO_NULL, MPI_COMM_WORLD, &win_d);

    MPI_Win_create(local_changed.data(), local_vertex_count * sizeof(long long),
                   sizeof(long long), MPI_INFO_NULL, MPI_COMM_WORLD, &win_changed);

    unordered_map<long long, set<int>> buckets;

    int starting_point = local_to_global_index(0, rank, num_vertices, num_procs);

    for (int i = 0; i < local_vertex_count; ++i) {
        int global_id = starting_point + i;
        if (global_id != root) {
            buckets[INF/delta].insert(global_id);
        }
    }

    if (owner(root, num_vertices, num_procs) == rank) {
        int li = global_to_local_index(root, rank, num_vertices, num_procs);
        local_d[li] = 0;
        local_d_prev[li] = 0;
        buckets[0].insert(root);
    }

    MPI_Barrier(MPI_COMM_WORLD); // Ensure window is ready

    long long k = 0;
    bool continue_running = true;

    long long local_processed_vertices = 0;
    long long total_processed_vertices = 0;

    while (continue_running) {
        set<int>  set_of_processed_vertices;
        set<int> A = buckets[k];

        bool filled_buckets = 0;
        for (const auto& b : buckets) {
            filled_buckets |= (!b.second.empty());
        }

        MPI_Allreduce(&filled_buckets, &continue_running, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
        if (!continue_running) break;

        bool local_flag = true, global_flag = true;

        while (global_flag) {
            // All vertices in A will be processed within this bucket
            set_of_processed_vertices.insert(A.begin(), A.end());

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

        local_processed_vertices+= set_of_processed_vertices.size();
        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Allreduce(&local_processed_vertices, &total_processed_vertices, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        // // cout << "Here I am at rank" << rank <<" my k is: " << k << endl;

        buckets[k].clear();
        k += 1;
        MPI_Barrier(MPI_COMM_WORLD);


        if (total_processed_vertices > (tau * num_vertices)) {
            // process everything from other buckets at once;
            A.clear();

            set<int> result;
            global_flag = true;


            for (auto& it : buckets) {
                long long key = it.first;
                if (key >= k) {
                    result.insert(buckets[key].begin(), buckets[key].end());
                    
                }
            }
            // cout << "result rank: " << rank << " set size: " << result.size()  <<endl;

            for (auto it: result) {
                // cout << it << " ";
            }
            // cout << endl;

            while (global_flag) {

                if (rank ==3) {
                    // cout << "local_d: " << local_d.size() << " rank: " << rank << endl;
                    for (long long v: local_d) {
                        // cout << v << " ";
                    }
                    // cout << endl;
                }

                if (rank ==3) {
                    // cout << "result: " << result.size() << " rank: " << rank << endl;
                    for (long long v: result) {
                        // cout << v << " ";
                    }
                    // cout << endl;
                }

                MPI_Barrier(MPI_COMM_WORLD);

                set<int> current = result;
                process_bucket(current, vertex_mapping, rank, num_vertices, num_procs,
                            local_d, local_changed, local_d_prev, win_d, win_changed);

                if (rank ==3) {
                    // cout << "local_d: " << local_d.size() << " rank: " << rank << endl;
                    for (long long v: local_d) {
                        // cout << v << " ";
                    }
                    // cout << endl;
                }

                if (rank ==3) {
                    // cout << "local_changed: " << local_changed.size() << " rank: " << rank << endl;
                    for (long long v: local_changed) {
                        // cout << v << " ";
                    }
                    // cout << endl;
                }

                MPI_Barrier(MPI_COMM_WORLD);


                set<int> A_prim = update_set_and_collect_active(
                    local_d, local_changed, local_d_prev, current,
                    rank, num_vertices, num_procs
                );

                if (rank ==3) {
                    // cout << "local_d: " << local_d.size() << " rank: " << rank << endl;
                    for (long long v: local_d) {
                        // cout << v << " ";
                    }
                    // cout << endl;
                }

                if (rank ==3) {
                    // cout << "current: " << current.size() << " rank: " << rank << endl;
                    for (long long v: current) {
                        // cout << v << " ";
                    }
                    // cout << endl;
                }

                result.clear();
                set_intersection(A_prim.begin(), A_prim.end(), current.begin(), current.end(),
                                inserter(result, result.begin()));
                
                MPI_Barrier(MPI_COMM_WORLD);

                if (rank == 3) {
                                 // cout << "Result.size: " << result.size() << " rank: " << rank << endl;
                    for (int i = 0; i< local_vertex_count; i++){
                        // cout <<local_changed[i] << " ";
                    }
                    // cout << endl;
                }

                local_flag = !result.empty();
                MPI_Barrier(MPI_COMM_WORLD);

                MPI_Allreduce(&local_flag, &global_flag, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
            }
            break;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Win_free(&win_d);
    MPI_Win_free(&win_changed);

    unordered_map<int, long long> result;
    starting_point = local_to_global_index(0, rank, num_vertices, num_procs);

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

    // cout << "My rank is: " << rank << "\n";

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
    long long max_weight = 0;

    while (infile >> u >> v >> w) {

        if (u >= start_vertex && u <= end_vertex) {
            Edge e;
            e.v1 = u;
            e.v2 = v;
            e.weight = w;
            my_vertices[u].edges.push_back(e);
            // wystarczy tylko w 1 stronę krawedz
            max_weight = update_weight(max_weight, w);
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

    for (int i = start_vertex; i <= end_vertex; i++) {
        my_vertices[i].degree = my_vertices[i].edges.size();
    }

    cout << "Processing" << endl;
    unordered_map<int, long long> final_values = delta_stepping_basic(my_vertices, global_root, rank, num_processes, num_vertices);
    // unordered_map<int, long long> final_values = delta_stepping_prunning(my_vertices, global_root, rank, num_processes, num_vertices, max_weight);
    // unordered_map<int, long long> final_values = delta_stepping_hybrid(my_vertices, global_root, rank, num_processes, num_vertices);

    // Dummy output for testing (write -1 as shortest path for each vertex)
    std::ofstream outfile(output_file);
    if (!outfile.is_open()) {
        std::cerr << "Rank " << rank << ": Failed to open output file " << output_file << "\n";
        MPI_Finalize();
        return 1;
    }

    // outfile << "shit 5" << "\n";

    // for (int v = start_vertex; v <= end_vertex; ++v) {
    //     outfile << v << " edges:";
    //     Vertex vertex = my_vertices[v];
    //     for (Edge e: vertex.edges) {
    //         outfile << e.v2 << " w: " << e.weight << ";";
    //     }
    //     outfile << "\n";
    // }
    for (int v = start_vertex; v <= end_vertex; ++v) {
        // outfile << v << " " << final_values[v] << "\n";
        outfile << final_values[v] << "\n";
   
    }
    outfile.close();

    MPI_Finalize();
    return 0;
}
