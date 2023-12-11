import csv
import networkx as nx
import matplotlib.pyplot as plt

# Read CSV file and create a directed graph
def create_graph_from_csv(csv_file):
    graph = nx.DiGraph()
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            node_a, node_b, min_max_normalized, frequency = row
            for _ in range(int(frequency)):
                graph.add_edge(node_a, node_b)
            #graph.add_edge(node_a, node_b, weight=float(frequency))  # Assuming the weight is a float
    return graph

# Visualize node degree distribution
def visualize_degree_distribution(graph):
    degrees = dict(graph.degree())  # Get degrees of nodes

    degree_values = list(degrees.values())
    unique_degrees = list(set(degree_values))
    unique_degrees.sort()

    degree_counts = [degree_values.count(deg) for deg in unique_degrees]

    plt.figure(figsize=(10, 3))
    plt.scatter(unique_degrees, degree_counts, s=5)
    plt.xlabel('Degree')
    plt.ylabel('Number of Nodes')
    #plt.ylim([0, 1000])
    plt.title('Node Degree Distribution')
    plt.savefig('degree.png')

# Example usage
if __name__ == "__main__":
    input_csv_file = 'output.csv'

    graph = create_graph_from_csv(input_csv_file)
    visualize_degree_distribution(graph)