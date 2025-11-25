import osmnx as ox
import math
import heapq

def dijkstra(graph, source, target):
    dist = {node: float("inf") for node in graph.nodes()}
    dist[source] = 0

    prev = {node: None for node in graph.nodes()}

    pq = [(0, source)]
    visited = set()

    while pq:
        current_dist, u = heapq.heappop(pq)

        if u in visited:
            continue
        visited.add(u)

        if u == target:
            break

        for v in graph.neighbors(u):
            edge_data = list(graph[u][v].values())[0]
            weight = edge_data.get("length", 1)

            alt = current_dist + weight
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(pq, (alt, v))

    return dist, prev


def reconstruct_path(prev,source,target):
    path=[]
    node=target
    while node is not None:
        path.append(node)
        if node== source:
            break
        node=prev[node]
    return path[::-1]

place_point = (27.6748, 85.4274) 
print("Downloading map data...")

Graph=ox.graph_from_point(place_point,dist=1000,network_type='drive')

nodes_list=list(Graph.nodes())
start_node=nodes_list[1]

end_node=nodes_list[-1]

print(f"Calculating shortest path from Node {start_node} to Node {end_node}...")

dist, prev = dijkstra(Graph, start_node, end_node)
path = reconstruct_path(prev, start_node, end_node)

print("Shortest distance:", dist[end_node])
print("Path:", path)

fig, ax = ox.plot_graph_route(Graph, path, route_color='red', route_linewidth=3, node_size=0)