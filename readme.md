Dijkstra Shortest Path — Real Map Implementation
A Python implementation of Dijkstra's shortest path algorithm applied to real-world road network data using OpenStreetMap.
What it does

Downloads live road network data for a given location using osmnx
Builds a weighted graph where edge weights represent road segment lengths
Runs a custom Dijkstra implementation using a min-heap priority queue
Reconstructs and visualizes the shortest path on the actual map

Tech Stack

Python — core implementation
OSMnx — OpenStreetMap data retrieval and graph construction
NetworkX — graph structure
Heapq — min-heap for efficient priority queue
Matplotlib — route visualization

How to Run
bash# Install dependencies
pip install osmnx networkx matplotlib

Run the script
python DijkstraImpl.py
Key Concepts

Graph representation — road intersections as nodes, road segments as weighted edges
Priority queue — min-heap ensures the shortest tentative distance is always processed first
Path reconstruction — backtracking via a prev dictionary from target to source

Sample Output
Calculates shortest driving route between two nodes in a 1km radius around coordinates (27.6748, 85.4274) and plots the result with the route highlighted in red.
