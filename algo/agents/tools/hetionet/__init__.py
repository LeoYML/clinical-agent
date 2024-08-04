import json
from turtle import distance
import pandas as pd
from tqdm import tqdm

import networkx as nx
import pickle 
import os
import sys

cwd_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{cwd_path}/../../')

from utils import match_name, LOGGER, find_least_levenshtein_distance

# Generate NetworkX graph
if os.path.exists(f'{cwd_path}/data/nx_graph.pkl'):
    G = pickle.load(open(f'{cwd_path}/data/nx_graph.pkl', 'rb'))
else:
    LOGGER.log_with_depth("Data not found. Generating NetworkX graph...")
    # Read Hetionet v1.0
    fpath = 'data/hetionet-v1.0.json'

    with open(fpath, 'r') as f:
        hetio_json = json.load(f)
    
    # Nodes
    node_rows = []
    for idx, node in enumerate(hetio_json['nodes']):
        node_kind = node['kind'].lower()
        node_id = node['identifier']
        node_name = node['name'].lower()
        
        node_rows.append([node_kind, node_id, node_name])

    node_df = pd.DataFrame(node_rows, columns=['kind', 'id', 'name'])

    LOGGER.log_with_depth(node_df)
    node_df.to_csv(f"{cwd_path}/data/nodes.csv", sep='\t', index=False)

    LOGGER.log_with_depth(f"Node Kind: {node_df['kind'].value_counts()}")

    # Edges
    edge_rows = []
    filter_kind = ['Side Effect', 'Compound', 'Symptom', 'Anatomy', 'Pharmacologic Class', 'Disease']

    for idx, edge in enumerate(hetio_json['edges']):
        edge_kind = edge['kind'].lower()
        edge_source_kind, edge_source_id = edge['source_id']
        edge_target_kind, edge_target_id = edge['target_id']
        edge_direction = edge['direction']

        if edge_source_kind not in filter_kind or edge_target_kind not in filter_kind:
            continue

        edge_source_kind, edge_target_kind = edge_source_kind.lower(), edge_target_kind.lower()
        
        edge_rows.append([edge_kind, edge_source_kind, edge_source_id, edge_target_kind, edge_target_id, edge_direction])

    edge_df = pd.DataFrame(edge_rows, columns=['kind', 'source_kind', 'source_id', 'target_kind', 'target_id', 'direction'])
    LOGGER.log_with_depth(edge_df)
    edge_df.to_csv(f"{cwd_path}/data/edges.csv", sep='\t', index=False)

    # Original: 2250197
    # After filter: 155106

    # Create NetworkX graph
    get_name = lambda node_kind, node_id: node_df[(node_df['kind'] == node_kind) & (node_df['id'] == node_id)]['name'].values[0]

    G = nx.Graph()

    for idx, node in node_df.iterrows():
        G.add_node(node['name'], kind=node['kind'])

    for idx, edge in tqdm(edge_df.iterrows(), total=edge_df.shape[0]):
        source_name = get_name(edge['source_kind'], edge['source_id'])
        target_name = get_name(edge['target_kind'], edge['target_id'])
        
        G.add_edge(source_name, target_name, kind=edge['kind'])
    
    with open(f'{cwd_path}/data/nx_graph.pkl', 'wb') as f:
        pickle.dump(G, f)


def retrieval_hetionet(source_name, target_name, cutoff=2):
    try:
        # Function to list all paths with length < 3 (cutoff = 2)
        source_name, target_name = source_name.strip().lower(), target_name.strip().lower()

        # Match similar names
        source_name = match_name(source_name, G.nodes)
        target_name, distance = find_least_levenshtein_distance(target_name, G.nodes)

        all_paths = list(nx.all_simple_paths(G, source=source_name, target=target_name, cutoff=cutoff))

        final_path_str = f"All paths from {source_name} to {target_name} with length < {cutoff+1}:\n"

        path_list = []
        for path in all_paths:
            single_path = []
            for i in range(len(path) - 1):
                start_node = path[i]
                end_node = path[i + 1]
                
                edge_kind = G[start_node][end_node].get('kind', 'Unknown relation')
                start_node_kind = G.nodes[start_node].get('kind', 'Unknown kind')
                end_node_kind = G.nodes[end_node].get('kind', 'Unknown kind')

                if i == 0:
                    single_path.append(f"<drug>{start_node_kind}:{start_node}</drug>")
                
                single_path.append(f"<edge>{edge_kind}</edge><drug>{end_node_kind}:{end_node}</drug>")
            path_list.append(f"<path>{''.join(single_path)}</path>")
        
        if len(path_list) == 0:
            return ""
        else:
            final_path_str += '\n'.join(path_list)

            return final_path_str
    except Exception as e:
        LOGGER.log_with_depth(f"Warning: {e}")
        return ""

if __name__ == '__main__':
    start_node_name = "Escitalopram"
    end_node_name = "bipolar disorder"

    results = retrieval_hetionet(start_node_name, end_node_name)
    LOGGER.log_with_depth(results)