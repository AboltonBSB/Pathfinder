import requests
import helpers.visual_helpers as visualizers
from helpers.graph_helpers import build_edges

BASE_URL = "http://localhost:8000"

def compute_all_trees(grouped_nodes):
    """Trigger computation and storage for all groups."""
    for group_key in grouped_nodes.keys():
        print(f"Computing {group_key}...")
        response = requests.post(f"{BASE_URL}/compute/eigenvectors/{group_key}")
        print(response.json())

def visualize_tree(group_key, grouped_nodes):
    """Fetch stored results and visualize."""
    response = requests.get(f"{BASE_URL}/results/eigenvectors/{group_key}")

    if "error" in response.json():
        print(f"No results found for {group_key} — computing first...")
        requests.post(f"{BASE_URL}/compute/decomposition/{group_key}")
        response = requests.get(f"{BASE_URL}/results/eigenvectors/{group_key}")

    data = response.json()

    edges        = [tuple(e) for e in data["edges"]]       # JSON arrays back to tuples
    eigenvectors = data["eigenvectors"]
    node_group   = grouped_nodes[group_key]

    print(f"Number of eigenvectors returned: {len(eigenvectors)}")
    print(f"Length of first eigenvector: {len(eigenvectors[0]) if len(eigenvectors) > 0 else 'empty'}")

    # Rebuild reverse_map from stored node_count
    _, _, _, reverse_map = build_edges(node_group)

    visualizers.plot_fiedler_vector(edges, eigenvectors[1], node_group, reverse_map, title=group_key)
    visualizers.plot_spectral_embedding_3d(eigenvectors, 1, 2, 3, node_group, reverse_map, title=f"{group_key} 3D")

def compute_clustering(group_key):

    print(f"Computing clusters for {group_key}...")
    response = requests.post(f"{BASE_URL}/compute/clustering/{group_key}")
    #print(response.json())

def visualize_clustering(group_key, grouped_nodes):

    response_eigenvectors = requests.get(f"{BASE_URL}/results/eigenvectors/{group_key}")
    response_clusters = requests.get(f"{BASE_URL}/results/clustering/{group_key}")

    if "error" in response_eigenvectors.json():
        print(f"No results found for {group_key} — computing eigenvectors first...")
        requests.post(f"{BASE_URL}/compute/decomposition/{group_key}")
        response_eigenvectors = requests.get(f"{BASE_URL}/results/eigenvectors/{group_key}")

    if "error" in response_clusters.json():
        print(f"No clustering results found for {group_key} — computing clusters first...")
        requests.post(f"{BASE_URL}/compute/clustering/{group_key}")
        response_clusters = requests.get(f"{BASE_URL}/results/clustering/{group_key}")
    
    data_eigenvectors = response_eigenvectors.json()
    data_clusters = response_clusters.json()

    eigenvectors = data_eigenvectors["eigenvectors"]
    clusters = data_clusters["clusters"]
    node_group = grouped_nodes[group_key]
    _, _, _, reverse_map = build_edges(node_group)

    # 2D plot, not using PCA
    visualizers.plot_spectral_clusters_2d(
        eigenvectors,
        clusters,
        node_group,
        reverse_map,
        title=f"{group_key} Clusters"
    )
    # 3D plot, using PCA
    visualizers.plot_spectral_clusters_3d(
        eigenvectors,
        clusters,
        node_group,
        reverse_map,
        title=f"{group_key} Clusters 3D"
    )



if __name__ == "__main__":
    import json
    from pathlib import Path
    from collections import defaultdict
    from helpers.graph_helpers import build_grouped_nodes  

    # Load data
    base_path = Path(__file__).parent
    file_path = base_path / "../assets/data.json"
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    grouped_nodes = build_grouped_nodes(data)

    group_key = "main"

    requests.delete(f'{BASE_URL}/results/eigenvectors/{group_key}')
    requests.delete(f'{BASE_URL}/results/clustering/{group_key}')
    print(f"results reset, recomputing data for {group_key}")

    # Option A: compute everything fresh
    #compute_all_trees(grouped_nodes)
    requests.post(f'{BASE_URL}/compute/eigenvectors/{group_key}')

    # Option B: just visualize (uses cached results if available)
    #visualize_tree("main", grouped_nodes)
    #visualize_tree("Necromancer", grouped_nodes)
    compute_clustering("main")
    #visualize_clustering("Necromancer", grouped_nodes)
    visualize_clustering("main", grouped_nodes)





""" import json
from pathlib import Path
from collections import defaultdict
from helpers.visual_helpers import *
from helpers.graph_helpers import build_edges

import pyo3_example

base_path = Path(__file__).parent

file_path = base_path / "../assets/data.json"

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

ascendancyClasses = ['Slayer', 'Gladiator', 'Champion', 'Assassin', 'Saboteur', 'Trickster', 
'Juggernaut', 'Berserker', 'Chieftain', 'Necromancer', 'Occultist', 'Elementalist', 'Deadeye',
'Warden', 'Pathfinder', 'Inquisitor', 'Hierophant', 'Guardian', 'Ascendant', 'Reliquarian']

grouped_nodes = defaultdict(dict)

for node_id, node_data in data["nodes"].items():
    if node_data.get("out", []) == [] and node_data.get("in", []) == []:
        continue
    group_key = node_data.get("ascendancyName", "main")
    if (group_key in ascendancyClasses or group_key == 'main'):
        grouped_nodes[group_key][node_id] = node_data

print(f"found groups: {list(grouped_nodes.keys())}")
for key, nodes in grouped_nodes.items():
    print(f"{key}: {len(nodes)} nodes")

for group_key, node_group in grouped_nodes.items():
    edges, node_count, node_weights, reverse_id_map = build_edges(node_group)

    print(f"\nProcessing '{group_key}': {node_count} nodes, {len(edges)} edges")

    witch_ascendencies = ["Necromancer", "Occultist", "Elementalist"]

    if group_key == "Necromancer":
        (eigenvectors, eigenvalues) = pyo3_example.compute_laplacian(edges, node_weights)
        print(eigenvectors[1])

        plot_fiedler_vector(edges, eigenvectors[1], node_group, reverse_id_map)
        plot_spectral_embedding_2d(eigenvectors, 1, 2, node_group, reverse_id_map)
        plot_spectral_embedding_2d(eigenvectors, 2, 3, node_group, reverse_id_map)
        plot_spectral_embedding_3d(eigenvectors, x_index=1, y_index=2, z_index=3, node_group=node_group, reverse_map=reverse_id_map,title="Main Tree 3D Spectral Embedding") """


#pyo3_example.compute_laplacian(edges)