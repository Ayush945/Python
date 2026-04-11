# Dijkstra Shortest Path — Real Map Implementation

A Python implementation of Dijkstra's shortest path algorithm applied to real-world road network data using OpenStreetMap.

---

## What It Does

- Downloads live road network data for a given location using `osmnx`
- Builds a weighted graph where edge weights represent actual road segment lengths
- Runs a custom Dijkstra implementation using a min-heap priority queue
- Reconstructs and visualizes the shortest driving path on the real map

---

## Tech Stack

| Library | Purpose |
|---|---|
| `osmnx` | OpenStreetMap data retrieval and graph construction |
| `networkx` | Graph structure and node/edge management |
| `heapq` | Min-heap priority queue for Dijkstra |
| `matplotlib` | Route visualization |

---

## How to Run

```bash
# Install dependencies
pip install osmnx networkx matplotlib

# Run the script
python DijkstraImpl.py
```

---

## Key Concepts

- **Graph representation** — road intersections are nodes, road segments are weighted edges
- **Priority queue** — min-heap ensures the node with the shortest tentative distance is always processed first
- **Path reconstruction** — backtracks through a `prev` dictionary from target to source to recover the full route

---

## Sample Output

Calculates the shortest driving route between two nodes within a 1km radius of coordinates `(27.6748, 85.4274)` and plots the result with the route highlighted in red.
