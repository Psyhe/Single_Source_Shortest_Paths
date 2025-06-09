#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdbool.h>

typedef struct {
    int target;
    int weight;
} Edge;

typedef struct {
    int vertex;
    int dist;
} QueueNode;

typedef struct {
    QueueNode* data;
    int size;
    int capacity;
} MinHeap;

void swap(QueueNode* a, QueueNode* b) {
    QueueNode tmp = *a;
    *a = *b;
    *b = tmp;
}

MinHeap* create_min_heap(int capacity) {
    MinHeap* heap = malloc(sizeof(MinHeap));
    heap->data = malloc(capacity * sizeof(QueueNode));
    heap->size = 0;
    heap->capacity = capacity;
    return heap;
}

void heapify_up(MinHeap* heap, int i) {
    while (i > 0 && heap->data[(i - 1) / 2].dist > heap->data[i].dist) {
        swap(&heap->data[i], &heap->data[(i - 1) / 2]);
        i = (i - 1) / 2;
    }
}

void heapify_down(MinHeap* heap, int i) {
    int left = 2 * i + 1, right = 2 * i + 2, smallest = i;
    if (left < heap->size && heap->data[left].dist < heap->data[smallest].dist) smallest = left;
    if (right < heap->size && heap->data[right].dist < heap->data[smallest].dist) smallest = right;
    if (smallest != i) {
        swap(&heap->data[i], &heap->data[smallest]);
        heapify_down(heap, smallest);
    }
}

void heap_push(MinHeap* heap, int vertex, int dist) {
    heap->data[heap->size++] = (QueueNode){vertex, dist};
    heapify_up(heap, heap->size - 1);
}

QueueNode heap_pop(MinHeap* heap) {
    QueueNode min = heap->data[0];
    heap->data[0] = heap->data[--heap->size];
    heapify_down(heap, 0);
    return min;
}

bool heap_empty(MinHeap* heap) {
    return heap->size == 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) fprintf(stderr, "Usage: %s <input> <output>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    FILE* input = fopen(argv[1], "r");
    if (!input) {
        perror("fopen input");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int n_vertices, first_vertex, last_vertex;
    fscanf(input, "%d %d %d", &n_vertices, &first_vertex, &last_vertex);
    int local_n = last_vertex - first_vertex + 1;

    // Adjacency list
    int max_edges = 100000;
    Edge** adj = calloc(local_n, sizeof(Edge*));
    int* edge_counts = calloc(local_n, sizeof(int));
    int* edge_capacity = calloc(local_n, sizeof(int));
    for (int i = 0; i < local_n; ++i) {
        edge_capacity[i] = 8;
        adj[i] = malloc(edge_capacity[i] * sizeof(Edge));
    }

    int u, v, w;
    while (fscanf(input, "%d %d %d", &u, &v, &w) == 3) {
        if (u >= first_vertex && u <= last_vertex) {
            int idx = u - first_vertex;
            if (edge_counts[idx] >= edge_capacity[idx]) {
                edge_capacity[idx] *= 2;
                adj[idx] = realloc(adj[idx], edge_capacity[idx] * sizeof(Edge));
            }
            adj[idx][edge_counts[idx]++] = (Edge){v, w};
        }
        if (v >= first_vertex && v <= last_vertex) {
            int idx = v - first_vertex;
            if (edge_counts[idx] >= edge_capacity[idx]) {
                edge_capacity[idx] *= 2;
                adj[idx] = realloc(adj[idx], edge_capacity[idx] * sizeof(Edge));
            }
            adj[idx][edge_counts[idx]++] = (Edge){u, w};
        }
    }
    fclose(input);

    int* dist = malloc(n_vertices * sizeof(int));
    bool* visited = calloc(n_vertices, sizeof(bool));
    for (int i = 0; i < n_vertices; ++i) dist[i] = INT_MAX;
    if (rank == 0) dist[0] = 0;

    MinHeap* heap = create_min_heap(n_vertices);
    if (rank == 0 && 0 >= first_vertex && 0 <= last_vertex)
        heap_push(heap, 0, 0);

    while (1) {
        int local_min_dist = INT_MAX, local_min_vertex = -1;
        for (int i = 0; i < heap->size; ++i) {
            if (heap->data[i].dist < local_min_dist) {
                local_min_dist = heap->data[i].dist;
                local_min_vertex = heap->data[i].vertex;
            }
        }

        struct {
            int dist;
            int vertex;
        } local = {local_min_dist, local_min_vertex}, global;

        MPI_Allreduce(&local, &global, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);

        if (global.dist == INT_MAX) break;

        int u = global.vertex;
        visited[u] = true;
        if (u >= first_vertex && u <= last_vertex) {
            int local_idx = u - first_vertex;
            for (int i = 0; i < edge_counts[local_idx]; ++i) {
                int v = adj[local_idx][i].target;
                int weight = adj[local_idx][i].weight;
                if (!visited[v] && dist[u] + weight < dist[v]) {
                    dist[v] = dist[u] + weight;
                }
            }
        }

        // Broadcast updated distances to all
        MPI_Allreduce(MPI_IN_PLACE, dist, n_vertices, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        // Remove vertex from all heaps
        heap->size = 0;
        for (int i = first_vertex; i <= last_vertex; ++i) {
            if (!visited[i] && dist[i] < INT_MAX)
                heap_push(heap, i, dist[i]);
        }
    }

    FILE* output = fopen(argv[2], "w");
    if (!output) {
        perror("fopen output");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = first_vertex; i <= last_vertex; ++i)
        fprintf(output, "%d\n", dist[i]);

    fclose(output);

    free(dist);
    free(visited);
    for (int i = 0; i < local_n; ++i) free(adj[i]);
    free(adj);
    free(edge_counts);
    free(edge_capacity);
    free(heap->data);
    free(heap);

    MPI_Finalize();
    return 0;
}
