import os
import random
import heapq
import numpy as np
from collections import defaultdict

def generate_rmat_graph(n, edge_factor, A, B, C, D, weighted=True, max_weight=255):
    m = n * edge_factor
    edges = set()

    while len(edges) < m:
        u, v = 0, 0
        step = n // 2
        for _ in range(int(np.log2(n))):
            r = random.random()
            if r < A:
                pass  # u += 0, v += 0
            elif r < A + B:
                v += step
            elif r < A + B + C:
                u += step
            else:
                u += step
                v += step
            step //= 2
        if u != v:
            w = random.randint(0, max_weight) if weighted else 1
            edges.add((u, v, w))
            edges.add((v, u, w))  # undirected

    return list(edges)

def shard_vertices(n, k):
    base = n // k
    rem = n % k
    shards = []
    start = 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        end = start + size - 1
        shards.append((start, end))
        start = end + 1
    return shards

def write_input_files(n, k, edges, directory):
    os.makedirs(directory, exist_ok=True)
    adj = defaultdict(list)
    for u, v, w in edges:
        adj[u].append((v, w))

    shards = shard_vertices(n, k)
    for rank, (start_v, end_v) in enumerate(shards):
        path = os.path.join(directory, f'{rank}.in')
        with open(path, 'w') as f:
            f.write(f"{n} {start_v} {end_v}\n")
            for u in range(start_v, end_v + 1):
                for v, w in adj[u]:
                    f.write(f"{u} {v} {w}\n")

def read_inputs(n, k, directory):
    graph = defaultdict(list)
    rank_info = []
    for rank in range(k):
        path = os.path.join(directory, f'{rank}.in')
        with open(path) as f:
            lines = f.readlines()
            total_n, start_v, end_v = map(int, lines[0].split())
            rank_info.append((rank, start_v, end_v))
            for line in lines[1:]:
                u, v, w = map(int, line.strip().split())
                graph[u].append((v, w))
    return total_n, graph, rank_info

def dijkstra(n, graph, source=0):
    dist = [float('inf')] * n
    dist[source] = 0
    pq = [(0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    return dist

def write_output_files(rank_info, dist, directory):
    for rank, start_v, end_v in rank_info:
        path = os.path.join(directory, f'{rank}.out')
        with open(path, 'w') as f:
            for v in range(start_v, end_v + 1):
                f.write(f"{dist[v]}\n")

def run_tests_on_graph(n, edges, ks, tag):
    for k in ks:
        # Create full test directory name directly under "tests/"
        test_name = f"{tag}_k_{k}"
        test_dir = os.path.join("tests", test_name)

        os.makedirs(test_dir, exist_ok=True)

        write_input_files(n, k, edges, test_dir)
        total_n, graph, rank_info = read_inputs(n, k, test_dir)
        dist = dijkstra(total_n, graph)
        write_output_files(rank_info, dist, test_dir)

        print(f"✅ Finished test: {test_name} in '{test_dir}'")

def main():
    # n_values = [2 ** 9, 2 ** 10]
    # n_values = [2 ** 12]
    # n_values = [2 ** 13, 2**14]
    n_values = [2 ** 15]

    ks = [24, 48, 96]
    edge_factor = 16
    weighted = True

    for n in n_values:
        # RMAT-1: A=0.57, B=C=0.19, D=0.05
        print("🔹 Generating RMAT-1...")
        rmat1_edges = generate_rmat_graph(n, edge_factor, A=0.57, B=0.19, C=0.19, D=0.05, weighted=weighted)
        run_tests_on_graph(n, rmat1_edges, ks, tag=f"test_new_rmat1_n_{n}")

        # RMAT-2: A=0.5, B=C=0.1, D=0.3
        print("🔹 Generating RMAT-2...")
        rmat2_edges = generate_rmat_graph(n, edge_factor, A=0.5, B=0.1, C=0.1, D=0.3, weighted=weighted)
        run_tests_on_graph(n, rmat2_edges, ks, tag=f"test_new_rmat2_n_{n}")

if __name__ == "__main__":
    main()
