import pandas as pd
import networkx as nx
import community  # Louvain community detection
import random
import numpy as np
import matplotlib.pyplot as plt
import os

# Set a fixed seed for reproducibility
seed_value = 650
random.seed(seed_value)
np.random.seed(seed_value)

# Set the path to the folder containing CSV files
folder_path = 'D:\\preprocess_sem_5\\ICPE2024_DataChallenge_jan_27\\services_new\\'
folder_path1 = 'D:\\preprocess_sem_5\\ICPE2024_DataChallenge_jan_27\\louvain_new\\'

# Get a list of all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Loop through each CSV file
for csv_file in csv_files:
    # Construct the full path to the CSV file
    csv_file_path = os.path.join(folder_path, csv_file)

    df = pd.read_csv(csv_file_path)
    G_undirected = nx.from_pandas_edgelist(df, 'source', 'target', create_using=nx.Graph())

    # Select the largest connected component
    if nx.is_connected(G_undirected):
        largest_component = G_undirected
    else:
        largest_component_nodes = max(nx.connected_components(G_undirected), key=len)
        largest_component = G_undirected.subgraph(largest_component_nodes).copy()

    # Apply Louvain community detection on the largest component
    partition = community.best_partition(largest_component, random_state=seed_value)

    # Map partition back to the original DataFrame
    df['community'] = df['source'].map(partition)

    # Create a new DataFrame for the output format
    output_df = pd.DataFrame(columns=['timestamp', 'service', 'node', 'community', 'traceid'])
    output_df['time'] = df['timestamp']
    output_df['service'] = df['service']
    output_df['node'] = df['source']
    output_df['community'] = df['community']
    output_df['traceid'] = df['traceid']

    target_rows = pd.DataFrame(columns=['time', 'service', 'node', 'community', 'traceid'])
    target_rows['time'] = df['timestamp']
    target_rows['service'] = df['service']
    target_rows['node'] = df['target']
    target_rows['traceid'] = df['traceid']
    target_rows['community'] = df['target'].map(partition)
    output_df = pd.concat([output_df, target_rows])

    output_csv_path = os.path.join(folder_path1, f'{csv_file.replace("_output", "_result")}')
    output_df.to_csv(output_csv_path, index=False)

    # Create a layout for the largest component
    layout = nx.spring_layout(largest_component)

    # Draw the graph with nodes colored by community
    plt.figure(figsize=(12, 8))
    nx.draw(
        largest_component, 
        pos=layout, 
        node_color=[partition[node] for node in largest_component.nodes()],
        cmap=plt.cm.RdYlBu, 
        node_size=50, 
        with_labels=False
    )
    plt.title('Graph with Louvain Communities')
    node_labels = {node: node for node in largest_component.nodes()}
    nx.draw_networkx_labels(largest_component, pos=layout, labels=node_labels, font_size=8, font_color='black')
    
    # Save the plot as an image (adjust the filename as needed)
    plot_filename = os.path.join(folder_path1, f'{csv_file.replace("_output", "_plot")}.png')
    plt.savefig(plot_filename)
    plt.close()

     # Compute centrality metrics
    betweenness_centrality = nx.betweenness_centrality(largest_component)
    closeness_centrality = nx.closeness_centrality(largest_component)
    degree_centrality = nx.degree_centrality(largest_component)
    eigenvector_centrality = nx.eigenvector_centrality(largest_component, max_iter=1000)
    pagerank = nx.pagerank(largest_component)
    eccentricity = nx.eccentricity(largest_component)


    # Create DataFrame for node metrics
    centrality_df = pd.DataFrame({
        'node': list(largest_component.nodes()),
        'betweenness_centrality': [betweenness_centrality[node] for node in largest_component.nodes()],
        'closeness_centrality': [closeness_centrality[node] for node in largest_component.nodes()],
        'degree_centrality': [degree_centrality[node] for node in largest_component.nodes()],
        'eigenvector_centrality': [eigenvector_centrality[node] for node in largest_component.nodes()],
        'pagerank': [pagerank[node] for node in largest_component.nodes()],
        'eccentricity': [eccentricity[node] for node in largest_component.nodes()],
        'community': [partition[node] for node in largest_component.nodes()]
    })

    # Save node metrics to CSV
    centrality_csv_path = os.path.join(folder_path1, f'{csv_file.replace("_output", "_centrality")}.csv')
    centrality_df.to_csv(centrality_csv_path, index=False)

    # Calculate metrics for the graph
    metrics = {
        'number_of_nodes': largest_component.number_of_nodes(),
        'number_of_edges': largest_component.number_of_edges(),
        'density': nx.density(largest_component),
        'average_clustering': nx.average_clustering(largest_component),
        'average_degree': sum(dict(largest_component.degree()).values()) / largest_component.number_of_nodes(),
        'diameter': nx.diameter(largest_component) if nx.is_connected(largest_component) else "Not connected",
        'radius': nx.radius(largest_component) if nx.is_connected(largest_component) else "Not connected",
    }

    # Save metrics to a file
    metrics_filename = os.path.join(folder_path1, f'{csv_file.replace("_output", "_metrics")}.txt')
    with open(metrics_filename, 'w') as metrics_file:
        for key, value in metrics.items():
            metrics_file.write(f'{key}: {value}\n')
