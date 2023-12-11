import csv
import random
import networkx as nx

G = nx.DiGraph()
with open('output.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header if exists
    for row in reader:
        node_a, node_b, min_max_normalized, _ = row
        G.add_edge(node_a, node_b, weight=float(min_max_normalized))

# Function to generate random paths
def generate_random_paths(graph, num_paths):
    paths = []
    nodes = list(graph.nodes())
    while len(paths) < num_paths:
        start = random.choice(nodes)
        end = random.choice(nodes)
        while start == end:
            end = random.choice(nodes)
        try:
            path = nx.shortest_path(graph, source=start, target=end)
            paths.append(path)
        except nx.NetworkXNoPath:
            pass  # Handle the case where there is no path between the nodes
    return paths[:num_paths]

# Generate 10,000 paths
num_paths = 100000
all_paths = generate_random_paths(G, num_paths)

# Split paths into training, validation, and testing sets (80%, 10%, 10%)
random.shuffle(all_paths)
train_size = int(0.8 * num_paths)
val_test_size = int(0.1 * num_paths)

train_paths = all_paths[:train_size]
val_paths = all_paths[train_size:train_size + val_test_size]
test_paths = all_paths[train_size + val_test_size:]

# Function to write paths to a file
def write_paths_to_file(paths, filename):
    with open(filename, 'w') as file:
        for path in paths:
            line = ','.join(str(node) for node in path) + '\n'
            file.write(line)

# Write paths to separate files
write_paths_to_file(train_paths, 'twitter-train.txt')
write_paths_to_file(val_paths, 'twitter-val.txt')
write_paths_to_file(test_paths, 'twitter-test.txt')