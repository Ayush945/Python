import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

# 1. SETUP: Define the Location
# We download the street network for a specific radius around a point.
# I've chosen a coordinate in Kathmandu (near Thamel) for this example.
place_point = (27.7172, 85.3240) 
print("Downloading map data... (this may take a few seconds)")

# 'network_type="drive"' gets us roads meant for cars.
# This creates a Graph G where Nodes = Intersections, Edges = Roads.
G = ox.graph_from_point(place_point, dist=1000, network_type='drive')

# 2. PRE-PROCESSING: Add Edge Weights
# Dijkstra needs a 'weight' (cost) to minimize. 
# OSMnx automatically calculates 'length' (in meters) for edges.
# We interpret this as: Weight w(u,v) = distance between intersection u and v.

# 3. SELECT START & END POINTS
# Since we can't just click on the map in code, we pick two random nodes >
# from the graph to serve as our Source (S) and Target (T).
nodes_list = list(G.nodes())
start_node = nodes_list[0]        # Arbitrary starting point
end_node = nodes_list[10]        # Arbitrary destination point

print(f"Calculating shortest path from Node {start_node} to Node {end_node}...")

# 4. IMPLEMENTING DIJKSTRA (Using NetworkX)
# In your research project, you might write this logic from scratch using 'heapq'.
# For now, we use the library to understand the data structure.
try:
    shortest_path = nx.shortest_path(
        G, 
        source=start_node, 
        target=end_node, 
        weight='length',  # The attribute to minimize
        method='dijkstra'
    )
    print("Path found!")
    
    # Calculate total length of the path
    path_length = nx.path_weight(G, shortest_path, weight='length')
    print(f"Total Trip Distance: {path_length:.2f} meters")

    # 5. VISUALIZATION
    # This is the part that looks great in a portfolio/report.
    fig, ax = ox.plot_graph_route(
        G, 
        shortest_path, 
        route_color='r',       # Red path
        route_linewidth=6,     # Thick line
        bgcolor='k'            # Black background (looks pro)
    )

except nx.NetworkXNoPath:
    print("No path exists between these two nodes (graph might be disconnected).")