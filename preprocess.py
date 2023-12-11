import csv
import numpy as np
import pandas as pd
from collections import defaultdict


df_orig = pd.read_csv('clarin_plos_15072021.csv')

df = df_orig.dropna(subset=['retweeted_user_id'])


def minmax_normalize(data):
    min_val = min(data.values())
    max_val = max(data.values())

    normalized_data = {}
    for edge, freq in data.items():
        normalized_val = (freq - min_val) / (max_val - min_val)
        normalized_data[edge] = normalized_val
    
    return normalized_data


edge_weights = defaultdict(int)
for index, row in df.iterrows():
    edge = (int(row['user_id']), int(row['retweeted_user_id']))
    if edge in edge_weights:
        edge_weights[edge] += 1
    else:
        edge_weights[edge] = 1

normalized_frequencies = minmax_normalize(edge_weights)

with open('output.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['NodeA', 'NodeB', 'MinMaxNormalized', 'Frequency'])  # Writing the header row

    for edge, freq in edge_weights.items():
        normalized_val = normalized_frequencies[edge]
        writer.writerow([edge[0], edge[1], normalized_val, freq])