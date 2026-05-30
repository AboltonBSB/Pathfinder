from collections import defaultdict

ascendancyClasses = ['Slayer', 'Gladiator', 'Champion', 'Assassin', 'Saboteur', 'Trickster', 
'Juggernaut', 'Berserker', 'Chieftain', 'Necromancer', 'Occultist', 'Elementalist', 'Deadeye',
'Warden', 'Pathfinder', 'Inquisitor', 'Hierophant', 'Guardian', 'Ascendant', 'Reliquarian']

clusterJewels = ['Small Jewel Socket', 'Medium Jewel Socket', 'Large Jewel Socket']

def get_node_weight(node_data):
    if "isKeystone" in node_data:
        return 10.0
    if "isNotable" in node_data:
        return 5.0
    if "isMastery" in node_data:
        return 2.0
    else:
        return 1.0

def build_grouped_nodes(data):
    """Group nodes by ascendancy name, or 'main' if none."""
    grouped_nodes = defaultdict(dict)

    for node_id, node_data in data["nodes"].items():
        if "isMastery" in node_data:
            continue
        if node_id == "root":
            continue
        if node_data["name"] == "Position Proxy":
            continue
        if "isJewelSocket" in node_data and node_data["name"] in clusterJewels:
            continue
        if not node_data.get("out") and not node_data.get("in"):
            continue

        group_key = node_data.get("ascendancyName", "main")
        if (group_key in ascendancyClasses or group_key == "main"):
            grouped_nodes[group_key][node_id] = node_data

    return grouped_nodes

def build_edges(node_group):
    #all_node_ids = sorted(node_group.keys())
    id_map = {node_ids: i for i, node_ids in enumerate(node_group)}

    edge_set = set()
    for node_id, node_data in node_group.items():
        if "out" in node_data:
            for target_id in node_data["out"]:
                if target_id in id_map:
                    a, b = id_map[node_id], id_map[target_id]
                    edge_set.add((min(a,b), max(a, b)))
        if "in" in node_data:
            for source_id in node_data["in"]:
                if source_id in id_map:
                    a,b = id_map[node_id], id_map[source_id]
                    edge_set.add((min(a,b), max(a, b)))

    nodes_in_edges = {n for edge in edge_set for n in edge}
    connected = sorted(nodes_in_edges)
    remap = {old: new for new, old in enumerate(connected)}
    remapped_edges = [(remap[a], remap[b]) for a, b in edge_set]

    # Build weight list
    reverse_id_map = {i: node_id for node_id, i in id_map.items()}
    node_weights = []
    for new_idx in range(len(connected)):
        old_idx = connected[new_idx]
        node_id = reverse_id_map[old_idx]
        node_data = node_group[node_id]
        node_weights.append(get_node_weight(node_data))
    
    return remapped_edges, len(connected), node_weights, reverse_id_map

def detect_branches(edges, node_count):
    """
    Detect natural branches by walking from each leaf node
    back to the root, grouping nodes into branches.
    """

    G = nx.Graph()
    G.add_nodes_from(range(node_count))
    G.add_edges_from(edges)

    degrees = dict(G.degree())

    # Root = highest degree node (the ascendancy start node)
    root    = max(degrees, key=lambda n: degrees[n])

    # Leaves = degree 1 nodes
    leaves  = [n for n, d in degrees.items() if d == 1]

    print(f"Root: {root}")
    print(f"Leaves: {leaves}")
    print(f"Node degrees: {sorted(degrees.items())}")

    labels    = [-1] * node_count
    branch_id = 0

    # Assign root to cluster 0
    labels[root] = 0

    # For each leaf, trace the path back to root
    # Every node on that path gets the same branch label
    for leaf in leaves:
        path = nx.shortest_path(G, leaf, root)

        # Assign branch to all nodes on path except root
        for node in path[:-1]:  # exclude root
            if labels[node] == -1:  # only assign if not yet labeled
                labels[node] = branch_id + 1

        branch_id += 1

    # Any still unlabeled nodes are near-root connectors
    # assign them to root's cluster (0)
    for i in range(node_count):
        if labels[i] == -1:
            labels[i] = 0

    print(f"Branch labels: {labels}")
    print(f"Distribution: {Counter(labels)}")

    return labels