import networkx as nx
import matplotlib.pyplot as plt


Graph=nx.DiGraph()

people = ["Ayush","Ram","Shyam","Sita","Hari"]
Graph.add_nodes_from(people)

Graph.add_edge("Ayush","Ram",relationship="Friends")
Graph.add_edge("Ram","Ayush",relationship="Friends")

Graph.add_edge("Shyam","Hari",relationship="Friends")
Graph.add_edge("Hari","Shyam",relationship="Friends")

Graph.add_edge("Shyam","Sita",relationship="Friends")
Graph.add_edge("Sita","Shyam",relationship="Friends")

Graph.add_edge("Sita","Ram",relationship="wife")
#Graph.add_edge("Ram","Sita",relationship="wife")

Graph.add_edge("Ayush","Hari",relationship="Boss")

path_exists=nx.has_path(Graph, "Ayush","Sita")
print(f"Path from Ayush to Sita: {path_exists}")

if path_exists:
    shortest_path=nx.shortest_path(Graph, "Ayush", "Sita")
    print(f"Shortest path: {shortest_path}")

pos={
    "Ayush":(1,1),
    "Ram":(1,2),
    "Sita":(1,3),
    "Hari":(2,1),
    "Shyam":(2,2),
}

nx.draw(Graph, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold")

edge_labels=nx.get_edge_attributes(Graph,"relationship")
nx.draw_networkx_edge_labels(Graph,pos, edge_labels=edge_labels,font_color="red")

plt.title("Social Network Model")
plt.axis("off")
plt.show()
