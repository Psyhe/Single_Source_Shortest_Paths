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

int delta = 1; // TODO fine-tune

const int BASIC = 0;
const int IOS = 1;
const int PRUNING = 2;
const int HYBRID = 3;
const int OPT = 4;

struct Edge {
    int v1, v2;
    long long weight;
};

struct Vertex {
    int id;
    vector<Edge> edges;
    int degree;
};

long long get_bucket(long long distance, int delta) 
{
    return distance / delta;
}

long long update_weight(long long current_max, long long potential) {
    if (current_max >= potential) {
        return current_max;
    }
    else {
        return potential;
    }
}

long long get_global_min_bucket(unordered_map<long long, set<int>>& buckets) 
{
    long long local_min_bucket = INF;
    for (const auto& [idx, nodes] : buckets) 
    {
        if (!nodes.empty()) 
            local_min_bucket = min(local_min_bucket, idx);
    }
    long long global_min_bucket;
    MPI_Allreduce(&local_min_bucket, &global_min_bucket, 1, MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
    return global_min_bucket;
}

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

int should_continue_running(const unordered_map<long long, set<int>>& buckets) {
    int filled_buckets = 0;
    for (const auto& b : buckets) {
        filled_buckets |= (!b.second.empty());
    }

    int global_continue = 0;
    MPI_Allreduce(&filled_buckets, &global_continue, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    return global_continue;
}

unordered_map<long long, set<int>> initialize_buckets(
    int rank,
    int num_vertices,
    int num_procs,
    int root,
    int local_vertex_count,
    vector<long long>& local_d,
    vector<long long>& local_d_prev
) {
    unordered_map<long long, set<int>> buckets;

    int starting_point = local_to_global_index(0, rank, num_vertices, num_procs);
    for (int i = 0; i < local_vertex_count; i++) {
        int global_id = starting_point + i;
        if (global_id != root) {
            buckets[INF / delta].insert(global_id);
        }
    }

    if (owner(root, num_vertices, num_procs) == rank) {
        int li = global_to_local_index(root, rank, num_vertices, num_procs);
        local_d[li] = 0;
        local_d_prev[li] = 0;
        buckets[0].insert(root);
    }

    return buckets;
}

unordered_map<int, long long> gather_local_distances(
    const vector<long long>& local_d,
    int rank,
    int num_vertices,
    int num_procs
) {
    unordered_map<int, long long> result;
    int local_vertex_count = local_d.size();
    int starting_point = local_to_global_index(0, rank, num_vertices, num_procs);

    for (int i = 0; i < local_vertex_count; i++) {
        int global_id = starting_point + i;
        result[global_id] = local_d[i];
    }

    return result;
}

void try_update_distance_and_flag_changed(
    int owner_rank,
    int local_idx,
    long long new_d,
    MPI_Win& win_d
) {
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, owner_rank, 0, win_d);
    long long current_dv;
    MPI_Get(&current_dv, 1, MPI_LONG_LONG, owner_rank, local_idx, 1, MPI_LONG_LONG, win_d);
    MPI_Win_flush(owner_rank, win_d);

    if (new_d < current_dv) {
        MPI_Put(&new_d, 1, MPI_LONG_LONG, owner_rank, local_idx, 1, MPI_LONG_LONG, win_d);
        MPI_Win_flush(owner_rank, win_d);
    }
    MPI_Win_unlock(owner_rank, win_d);
}

void relax_edge(
    int u, Edge& e, int rank, int num_vertices, int num_procs,
    unordered_map<int, Vertex>& vertex_mapping,
    vector<long long>& local_d,  
    vector<long long>& local_d_prev,
    MPI_Win& win_d
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
        try_update_distance_and_flag_changed(owner_rank, local_idx, new_d, win_d );
    }
}

void relax_edge_IOS(
    int u, Edge& e, int rank, int num_vertices, int num_procs,
    unordered_map<int, Vertex>& vertex_mapping,
    vector<long long>& local_d,  
    vector<long long>& local_d_prev,
    MPI_Win& win_d,  long long k
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

    if ((new_d < d_v) && (new_d <= ((k+1) * delta - 1))) {
        try_update_distance_and_flag_changed(owner_rank, local_idx, new_d, win_d );
    }
}

void process_bucket(
    const set<int>& A, unordered_map<int, Vertex>& vertex_mapping,
    int rank, int num_vertices, int num_procs,
    vector<long long>& local_d,  
    vector<long long>& local_d_prev,
    MPI_Win& win_d
) {
    for (int u : A) {
        Vertex& current_vertex = vertex_mapping[u];
        for (Edge& e : current_vertex.edges) {
            relax_edge(u, e, rank, num_vertices, num_procs,
                       vertex_mapping, local_d,   local_d_prev,
                       win_d );
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

void process_bucket_first_phase_IOS(
    const set<int>& A, unordered_map<int, Vertex>& vertex_mapping,
    int rank, int num_vertices, int num_procs,
    vector<long long>& local_d,  
    vector<long long>& local_d_prev,
    MPI_Win& win_d, long long k
) {
    for (int u : A) {
        Vertex& current_vertex = vertex_mapping[u];
        for (Edge& e : current_vertex.edges) {
            if (e.weight <= delta) {
                relax_edge_IOS(u, e, rank, num_vertices, num_procs,
                        vertex_mapping, local_d,   local_d_prev,
                        win_d,   k);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

void process_bucket_outer_short(
    const set<int>& A, unordered_map<int, Vertex>& vertex_mapping,
    int rank, int num_vertices, int num_procs,
    vector<long long>& local_d,  
    vector<long long>& local_d_prev,
    MPI_Win& win_d
) {
    for (int u : A) {
        Vertex& current_vertex = vertex_mapping[u];
        for (Edge& e : current_vertex.edges) {
            if (e.weight <= delta) {
                relax_edge(u, e, rank, num_vertices, num_procs,
                        vertex_mapping, local_d,   local_d_prev,
                        win_d );
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

set<int> update_buckets_and_collect_active_set(
    vector<long long>& local_d,  
    vector<long long>& local_d_prev, unordered_map<long long, set<int>>& buckets,
    int rank, int num_vertices, int num_procs, long long k
) {
    set<int> A_prim;
    int local_vertex_count = local_d.size();

    for (int i = 0; i < local_vertex_count; i++) {
        if (local_d_prev[i] != local_d[i]) {
            long long old_bucket = get_bucket(local_d_prev[i], delta);
            long long new_bucket = get_bucket(local_d[i], delta);
            int global_id = local_to_global_index(i, rank, num_vertices, num_procs);

            if (new_bucket < old_bucket) {
                buckets[old_bucket].erase(global_id);
                buckets[new_bucket].insert(global_id);
            }

            if (local_d_prev[i] > local_d[i]) {
                A_prim.insert(global_id);
            }

            local_d_prev[i] = local_d[i];
        }
    }

    return A_prim;
}

set<int> update_set_and_collect_active(
    vector<long long>& local_d,  
    vector<long long>& local_d_prev, set<int>& current_set,
    int rank, int num_vertices, int num_procs
) {
    set<int> A_prim;
    int local_vertex_count = local_d.size();

    for (int i = 0; i < local_vertex_count; i++) {
        if (local_d_prev[i] != local_d[i]) {
            int global_id = local_to_global_index(i, rank, num_vertices, num_procs);
            if (local_d_prev[i] > local_d[i]) {
                current_set.insert(global_id);
            }

            if (local_d_prev[i] > local_d[i]) {
                A_prim.insert(global_id);
            }

            local_d_prev[i] = local_d[i];
        }
    }

    return A_prim;
}

void intersect_and_check_active_set(
    set<int>& A,
    unordered_map<long long, set<int>>& buckets,
    vector<long long>& local_d,
     
    vector<long long>& local_d_prev,
    int rank,
    int num_vertices,
    int num_procs,
    long long k,
    int& global_flag
) {
    set<int> A_prim = update_buckets_and_collect_active_set(
        local_d,   local_d_prev, buckets,
        rank, num_vertices, num_procs, k
    );

    A.clear();
    set_intersection(
        A_prim.begin(), A_prim.end(), buckets[k].begin(), buckets[k].end(),
        inserter(A, A.begin())
    );

    int local_flag = !A.empty();
    MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
}

void loop_process_bucket_phase(
    set<int>& A,
    unordered_map<int, Vertex>& vertex_mapping,
    unordered_map<long long, set<int>>& buckets,
    int rank,
    int num_vertices,
    int num_procs,
    vector<long long>& local_d,
     
    vector<long long>& local_d_prev,
    MPI_Win& win_d,
    long long k
) {
    int local_flag = 1, global_flag = 1;

    while (global_flag) {
        process_bucket(A, vertex_mapping, rank, num_vertices, num_procs,
                       local_d,   local_d_prev, win_d );

        intersect_and_check_active_set(
            A, buckets, local_d,   local_d_prev,
            rank, num_vertices, num_procs, k, global_flag
        );
    }
}

void loop_process_bucket_phase_ios(
    set<int>& A,
    unordered_map<int, Vertex>& vertex_mapping,
    unordered_map<long long, set<int>>& buckets,
    int rank,
    int num_vertices,
    int num_procs,
    vector<long long>& local_d,
     
    vector<long long>& local_d_prev,
    MPI_Win& win_d,
    long long k
) {
    int local_flag = 1, global_flag = 1;

    while (global_flag) {
        process_bucket_first_phase_IOS(
            A, vertex_mapping, rank, num_vertices, num_procs,
            local_d,   local_d_prev, win_d,   k
        );

        intersect_and_check_active_set(
            A, buckets, local_d,   local_d_prev,
            rank, num_vertices, num_procs, k, global_flag
        );
    }
}

void loop_process_bucket_outer_short_phase(
    set<int>& A,
    unordered_map<int, Vertex>& vertex_mapping,
    unordered_map<long long, set<int>>& buckets,
    int rank,
    int num_vertices,
    int num_procs,
    vector<long long>& local_d,
     
    vector<long long>& local_d_prev,
    MPI_Win& win_d,
    long long k
) {
    int local_flag = 1, global_flag = 1;

    while (global_flag) {
        process_bucket_outer_short(
            A, vertex_mapping, rank, num_vertices, num_procs,
            local_d,   local_d_prev, win_d 
        );

        intersect_and_check_active_set(
            A, buckets, local_d,   local_d_prev,
            rank, num_vertices, num_procs, k, global_flag
        );
    }
}

void loop_process_dynamic_bucket_phase_hybrid(
    set<int>& result,
    unordered_map<int, Vertex>& vertex_mapping,
    int rank,
    int num_vertices,
    int num_procs,
    vector<long long>& local_d,
     
    vector<long long>& local_d_prev,
    MPI_Win& win_d
) {
    int local_flag = 1, global_flag = 1;

    while (global_flag) {
        MPI_Barrier(MPI_COMM_WORLD);

        set<int> current = result;

        process_bucket(
            current, vertex_mapping, rank, num_vertices, num_procs,
            local_d,   local_d_prev, win_d 
        );

        MPI_Barrier(MPI_COMM_WORLD);

        set<int> A_prim = update_set_and_collect_active(
            local_d,   local_d_prev, current,
            rank, num_vertices, num_procs
        );

        result.clear();
        set_intersection(
            A_prim.begin(), A_prim.end(), current.begin(), current.end(),
            inserter(result, result.begin())
        );

        MPI_Barrier(MPI_COMM_WORLD);

        local_flag = !result.empty();
        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    }
}

long long local_push(
    const set<int>& current_bucket,
    const unordered_map<int, Vertex>& vertex_mapping,
    const vector<long long>& local_d,
    long long k,                
    double delta,
    int w_max                   
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
    long long k,               
    double delta,
    int w_max,
    int rank,
    int num_vertices,
    int num_procs                  
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

struct PullRequest {
    long long requester;
    long long responder;
    long long edge_weight;
};

struct PullResponse {
    long long node;
    long long d_v_prim;
};

void pull_model(
    unordered_map<int, Vertex>& vertex_mapping,
    vector<long long>& local_d,
    vector<long long>& local_d_prev,
    long long k,
    int rank,
    int num_procs,
    int num_vertices,
    unordered_map<long long, set<int>>& buckets,  
    int start_node,
    int end_node
) {
    vector<vector<PullRequest>> outgoing_requests(num_procs);

    for (int v = start_node; v <= end_node; ++v) {
        int local_idx = global_to_local_index(v, rank, num_vertices, num_procs);
        long long dv = local_d[local_idx];

        if (dv >= (k + 1) * delta) {
            Vertex current = vertex_mapping[v];
            for (const auto& edge : current.edges) {
                int u = edge.v2;
                long long w = edge.weight;

                if (w < dv - k * delta) {
                    int local_owner = owner(u, num_vertices, num_procs);
                    outgoing_requests[local_owner].push_back({v, u, w});
                }
            }
        }
    }

    // Flatten and send requests
    vector<int> sendcounts(num_procs), recvcounts(num_procs);
    vector<int> send_displs(num_procs), recv_displs(num_procs);
    vector<long long> sendbuf;


    for (int i = 0; i < num_procs; i++) {
        sendcounts[i] = outgoing_requests[i].size() * 3; // 3 = sizeof request struct
    }

    send_displs[0] = 0;
    for (int i = 1; i < num_procs; i++) {
        send_displs[i] = send_displs[i - 1] + sendcounts[i - 1];
    }

    int total_send = send_displs[num_procs - 1] + sendcounts[num_procs - 1];
    sendbuf.resize(total_send);

    for (int i = 0; i < num_procs; i++) {
        int offset = send_displs[i];
        for (const auto& req : outgoing_requests[i]) {
            sendbuf[offset++] = req.requester;
            sendbuf[offset++] = req.responder;
            sendbuf[offset++] = req.edge_weight;
        }
    }

    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    recv_displs[0] = 0;
    for (int i = 1; i < num_procs; i++) {
        recv_displs[i] = recv_displs[i - 1] + recvcounts[i - 1];
    }

    int total_recv = recv_displs[num_procs - 1] + recvcounts[num_procs - 1];
    vector<long long> recvbuf(total_recv);

    MPI_Alltoallv(sendbuf.data(), sendcounts.data(), send_displs.data(), MPI_LONG_LONG,
                  recvbuf.data(), recvcounts.data(), recv_displs.data(), MPI_LONG_LONG, MPI_COMM_WORLD);

    vector<vector<PullResponse>> outgoing_responses(num_procs);

    for (size_t i = 0; i < recvbuf.size(); i += 3) {
        long long requester = recvbuf[i];
        long long u = recvbuf[i + 1];
        long long w = recvbuf[i + 2];

        if (u < start_node || u > end_node) {
            continue;
        }

        int u_local = global_to_local_index(u, rank, num_vertices, num_procs);
        
        long long du = local_d[u_local];

        if (buckets[k].count(u) > 0) {
            long long new_dist = du + w;
            outgoing_responses[owner(requester, num_vertices, num_procs)].push_back({requester, new_dist});
        }
    }

    for (int i = 0; i < num_procs; i++) {
        sendcounts[i] = outgoing_responses[i].size() * 2; // size of response
    }

    send_displs[0] = 0;
    for (int i = 1; i < num_procs; i++) {
        send_displs[i] = send_displs[i - 1] + sendcounts[i - 1];
    }

    total_send = send_displs[num_procs - 1] + sendcounts[num_procs - 1];
    sendbuf.resize(total_send);

    for (int i = 0; i < num_procs; i++) {
        int offset = send_displs[i];
        for (const auto& res : outgoing_responses[i]) {
            sendbuf[offset++] = res.node;
            sendbuf[offset++] = res.d_v_prim;
        }
    }

    // Exchange responses
    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    recv_displs[0] = 0;
    for (int i = 1; i < num_procs; i++) {
        recv_displs[i] = recv_displs[i - 1] + recvcounts[i - 1];
    }

    total_recv = recv_displs[num_procs - 1] + recvcounts[num_procs - 1];
    recvbuf.resize(total_recv);

    MPI_Alltoallv(sendbuf.data(), sendcounts.data(), send_displs.data(), MPI_LONG_LONG,
                  recvbuf.data(), recvcounts.data(), recv_displs.data(), MPI_LONG_LONG, MPI_COMM_WORLD);


    for (size_t i = 0; i < recvbuf.size(); i += 2) {
        int v = recvbuf[i];
        long long new_dist = recvbuf[i + 1];
        if (v >= start_node && v <= end_node) {
            int v_local_index = global_to_local_index(v, rank, num_vertices, num_procs);

            long long old_dist = local_d[v_local_index];
            if (new_dist < old_dist) {
                int old_bucket_idx = get_bucket(old_dist, delta);
                int new_bucket_idx = get_bucket(new_dist, delta);

                if (buckets.count(old_bucket_idx)) {
                    buckets[old_bucket_idx].erase(v);
                }

                local_d_prev[v_local_index] = old_dist;
                local_d[v_local_index] = new_dist;

                buckets[new_bucket_idx].insert(v);
            }
        }
    }
}

void algorithm_ios(
    set<int>& A,
    unordered_map<int, Vertex>& vertex_mapping,
    unordered_map<long long, set<int>>& buckets,
    long long& k,
    int rank,
    int num_vertices,
    int num_procs,
    vector<long long>& local_d,
     
    vector<long long>& local_d_prev,
    MPI_Win& win_d
) {
    loop_process_bucket_phase_ios(
        A, vertex_mapping, buckets, rank, num_vertices, num_procs,
        local_d,   local_d_prev, win_d,   k
    );

    // Process long edges and outer short edges
    process_bucket(
        buckets[k], vertex_mapping, rank, num_vertices, num_procs,
        local_d,   local_d_prev, win_d 
    );

    set<int> A_prim = update_buckets_and_collect_active_set(
        local_d,   local_d_prev, buckets,
        rank, num_vertices, num_procs, k
    );

    buckets[k].clear();
}

void algorithm_pruning(
    set<int>& A,
    unordered_map<int, Vertex>& vertex_mapping,
    unordered_map<long long, set<int>>& buckets,
    long long& k,
    int rank,
    int num_vertices,
    int num_procs,
    vector<long long>& local_d,
     
    vector<long long>& local_d_prev,
    MPI_Win& win_d,
    long long delta,
    long long real_max_weight,
    int start_vertex,
    int end_vertex
) {
    loop_process_bucket_outer_short_phase(
        A, vertex_mapping, buckets, rank, num_vertices, num_procs,
        local_d,   local_d_prev, win_d,   k
    );

    long long local_push_count = local_push(
        buckets[k], vertex_mapping, local_d, k, delta, real_max_weight
    );
    long long local_pull_count = local_pull(
        buckets[k], vertex_mapping, local_d, k, delta, real_max_weight,
        rank, num_vertices, num_procs
    );

    long long total_push, total_pull;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(&local_push_count, &total_push, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_pull_count, &total_pull, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    if (total_pull < total_push) {

        pull_model(vertex_mapping, local_d,   local_d_prev, k, rank, num_procs, num_vertices, buckets, start_vertex, end_vertex);

        MPI_Barrier(MPI_COMM_WORLD);

    } else {
        process_bucket(
            buckets[k], vertex_mapping, rank, num_vertices, num_procs,
            local_d,   local_d_prev, win_d 
        );

        set<int> A_prim = update_buckets_and_collect_active_set(
            local_d,   local_d_prev, buckets,
            rank, num_vertices, num_procs, k
        );
    }

    buckets[k].clear();
}

int algorithm_hybrid(
    set<int>& A,
    unordered_map<int, Vertex>& vertex_mapping,
    unordered_map<long long, set<int>>& buckets,
    long long& k,
    int rank,
    int num_vertices,
    int num_procs,
    vector<long long>& local_d,
     
    vector<long long>& local_d_prev,
    MPI_Win& win_d,
    long long& local_processed_vertices,
    long long& total_processed_vertices
) {
    // All vertices in A will be processed within this bucket
    set<int>  set_of_processed_vertices;

    int global_flag = 1;
    int local_flag = 1;

    while (global_flag) {
        set_of_processed_vertices.insert(A.begin(), A.end());

        process_bucket(A, vertex_mapping, rank, num_vertices, num_procs,
                    local_d,   local_d_prev, win_d );

        set<int> A_prim = update_buckets_and_collect_active_set(
            local_d,   local_d_prev, buckets,
            rank, num_vertices, num_procs, k
        );

        A.clear();
        set_intersection(A_prim.begin(), A_prim.end(), buckets[k].begin(), buckets[k].end(),
                        inserter(A, A.begin()));

        local_flag = !A.empty();
        MPI_Allreduce(&local_flag, &global_flag, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
    }

    local_processed_vertices += set_of_processed_vertices.size();

    buckets[k].clear();

    MPI_Allreduce(&local_processed_vertices, &total_processed_vertices, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    if (total_processed_vertices > (tau * num_vertices)) {
        // Process everything from remaining buckets at once
        A.clear();
        set<int> result;

        for (auto& it : buckets) {
            if (it.first >= k) {
                result.insert(it.second.begin(), it.second.end());
            }
        }

        loop_process_dynamic_bucket_phase_hybrid(
            result, vertex_mapping, rank, num_vertices, num_procs,
            local_d,   local_d_prev, win_d 
        );
        return 1;
    }
    return 0;
}

int algorithm_opt(
    set<int>& A,
    unordered_map<int, Vertex>& vertex_mapping,
    unordered_map<long long, set<int>>& buckets,
    long long& k,
    int rank,
    int num_vertices,
    int num_procs,
    vector<long long>& local_d,
     
    vector<long long>& local_d_prev,
    MPI_Win& win_d,
    long long& local_processed_vertices,
    long long& total_processed_vertices,
    long long real_max_weight,
    int start_vertex,
    int end_vertex
) {
    // All vertices in A will be processed within this bucket
    set<int>  set_of_processed_vertices;

    int global_flag = 1;
    int local_flag = 1;

    // this part is from IOS - processing short inner edges first
    while (global_flag) {
        set_of_processed_vertices.insert(A.begin(), A.end());

        process_bucket_first_phase_IOS(
            A, vertex_mapping, rank, num_vertices, num_procs,
            local_d,   local_d_prev, win_d,   k
        );

        intersect_and_check_active_set(
            A, buckets, local_d,   local_d_prev,
            rank, num_vertices, num_procs, k, global_flag
        );
    }

    long long local_push_count = local_push(
        buckets[k], vertex_mapping, local_d, k, delta, real_max_weight
    );
    long long local_pull_count = local_pull(
        buckets[k], vertex_mapping, local_d, k, delta, real_max_weight,
        rank, num_vertices, num_procs
    );

    long long total_push, total_pull;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(&local_push_count, &total_push, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_pull_count, &total_pull, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    if (total_pull < total_push) {

        MPI_Barrier(MPI_COMM_WORLD);
        
        process_bucket_outer_short(buckets[k], vertex_mapping, rank, num_vertices, num_procs,
                    local_d,   local_d_prev, win_d );

        set<int> A_prim = update_buckets_and_collect_active_set(
            local_d,   local_d_prev, buckets,
            rank, num_vertices, num_procs, k
        );        
        MPI_Barrier(MPI_COMM_WORLD);

        pull_model(vertex_mapping, local_d,   local_d_prev, k, rank, num_procs, num_vertices, buckets, start_vertex, end_vertex);

        MPI_Barrier(MPI_COMM_WORLD);


    } else {
        // both short outer and long edges
        process_bucket(
            buckets[k], vertex_mapping, rank, num_vertices, num_procs,
            local_d,   local_d_prev, win_d 
        );

        set<int> A_prim = update_buckets_and_collect_active_set(
            local_d,   local_d_prev, buckets,
            rank, num_vertices, num_procs, k
        );
    }

    // Hybrid part
    local_processed_vertices += set_of_processed_vertices.size();

    buckets[k].clear();

    MPI_Allreduce(&local_processed_vertices, &total_processed_vertices, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    if (total_processed_vertices > (tau * num_vertices)) {
        A.clear();
        set<int> result;

        for (auto& it : buckets) {
            if (it.first >= k) {
                result.insert(it.second.begin(), it.second.end());
            }
        }

        loop_process_dynamic_bucket_phase_hybrid(
            result, vertex_mapping, rank, num_vertices, num_procs,
            local_d,   local_d_prev, win_d 
        );
        return 1;
    }
    return 0;
}

unordered_map<int, long long> algorithm(unordered_map<int, Vertex> vertex_mapping, int root, int rank, int num_procs, int num_vertices, long long max_weight, const int option, int start_vertex, int end_vertex) {
    // cout << "Beginning of processing algorithm" << endl;
    int local_vertex_count = vertices_for_rank(rank, num_vertices, num_procs);
    vector<long long> local_d(local_vertex_count, INF);
    vector<long long> local_d_prev(local_vertex_count, INF);

    MPI_Win win_d ;

    MPI_Win_create(local_d.data(), local_vertex_count * sizeof(long long),
                   sizeof(long long), MPI_INFO_NULL, MPI_COMM_WORLD, &win_d);

    unordered_map<long long, set<int>> buckets = initialize_buckets(
        rank, num_vertices, num_procs, root, local_vertex_count, local_d, local_d_prev
    );

    MPI_Barrier(MPI_COMM_WORLD); // Ensure window is ready

    long long real_max_weight = 0;
    MPI_Allreduce(&max_weight, &real_max_weight, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

    long long k = 0;
    int continue_running = 1;

    long long local_processed_vertices = 0;
    long long total_processed_vertices = 0;

    while (continue_running) {
        long long global_min_bucket = get_global_min_bucket(buckets);

        if (global_min_bucket == INF/delta) {
            break;
        }

        k = global_min_bucket;

        set<int> A = buckets[k];

        continue_running = should_continue_running(buckets);

        if (!continue_running) break;

        if (option == BASIC) {
            loop_process_bucket_phase(A, vertex_mapping, buckets, rank, num_vertices, num_procs,
                        local_d,   local_d_prev, win_d,   k);

            buckets[k].clear();
        }
        else if (option == IOS) {
            algorithm_ios(A, vertex_mapping, buckets, k, rank, num_vertices, num_procs,
                                    local_d,   local_d_prev, win_d);
        }
        else if (option == PRUNING) {
            algorithm_pruning(A, vertex_mapping, buckets, k, rank, num_vertices, num_procs,
                    local_d,   local_d_prev, win_d,   delta, real_max_weight, start_vertex, end_vertex);
        }
        else if (option == HYBRID) {
            int break_the_loop = algorithm_hybrid(
                A, vertex_mapping, buckets, k, rank, num_vertices, num_procs,
                local_d,   local_d_prev,
                win_d,  
                local_processed_vertices, total_processed_vertices
            );
            if (break_the_loop) {
                break;
            }
        }
        else if (option == OPT) {
            int break_the_loop = algorithm_opt(
                A, vertex_mapping, buckets, k, rank, num_vertices, num_procs,
                local_d,   local_d_prev,
                win_d,  
                local_processed_vertices, total_processed_vertices,
                real_max_weight, start_vertex, end_vertex
            );
            if (break_the_loop) {
                break;
            }        
        }
        else {
            cerr << "Incorrect opt" << endl;
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Win_free(&win_d);

    return gather_local_distances(local_d, rank, num_vertices, num_procs);
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
            // it is sufficient to check one of two directed edges
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

    // unordered_map<int, long long> final_values = algorithm(my_vertices, global_root, rank, num_processes, num_vertices, max_weight, BASIC, start_vertex, end_vertex);
    // unordered_map<int, long long> final_values = algorithm(my_vertices, global_root, rank, num_processes, num_vertices, max_weight, IOS, start_vertex, end_vertex);
    unordered_map<int, long long> final_values = algorithm(my_vertices, global_root, rank, num_processes, num_vertices, max_weight, PRUNING, start_vertex, end_vertex);
    // unordered_map<int, long long> final_values = algorithm(my_vertices, global_root, rank, num_processes, num_vertices, max_weight, HYBRID, start_vertex, end_vertex);
    // unordered_map<int, long long> final_values = algorithm(my_vertices, global_root, rank, num_processes, num_vertices, max_weight, OPT, start_vertex, end_vertex);

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
