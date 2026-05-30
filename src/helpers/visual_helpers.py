import numpy as np
import networkx as nx
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA

def plot_fiedler_vector(edges, fiedler_vector, node_group, reverse_map, title="Fiedler Vector on Graph"):
    fiedler = np.array(fiedler_vector)
    # Normalize to [-1, 1]
    fiedler_max = np.abs(fiedler).max()
    if fiedler_max > 0:  # guard against division by zero
        fiedler = fiedler / fiedler_max

    #Create graph from edge data
    G = nx.Graph()
    G.add_edges_from(edges)

    if len(edges) > 500:
        pos = nx.spectral_layout(G)
    else: 
        pos = nx.kamada_kawai_layout(G)

    """ #Build a lookup from the mapped index -> node name and data
    sorted_ids = sorted(node_group.keys())
    id_map = {node_id: i for i, node_id in enumerate(sorted_ids)}
    reverse_map = {i: node_id for node_id, i in id_map.items()} """

    node_list = sorted(G.nodes())

    hover_texts = []
    for n in node_list:
        node_id = reverse_map.get(n, "unknown")
        node_data = node_group.get(node_id, {})

        name = node_data.get("name", "Unknown")
        stats = node_data.get("stats", [])
        fiedler_val = fiedler[n]

        # Determine node type
        if "isKeystone" in node_data:
            node_type = "Keystone"
        elif "isNotable" in node_data:
            node_type = "Notable"
        elif "isMastery" in node_data:
            node_type = "Mastery"
        elif "isJewelSocket" in node_data:
            node_type = "Jewel Socket"
        else:
            node_type = "Regular"
        
        stats_text = "<br>".join(stats) if stats else "No stats"
        hover_texts.append(
            f"<b>{name}</b><br>"
            f"Type: {node_type}<br>"
            f"ID: {node_id}<br>"
            f"Fiedler: {fiedler_val:.6f}<br>"
            f"Stats:<br>{stats_text}"
        )
    
    x_nodes = [pos[n][0] for n in node_list]
    y_nodes = [pos[n][1] for n in node_list]
    fiedler_values = [fiedler[n] for n in node_list]

    x_edges, y_edges = [], []
    for a,b in edges:
        x_edges += [pos[a][0], pos[b][0], None]
        y_edges += [pos[a][1], pos[b][1], None]

    # Build the plot
    edge_trace = go.Scatter(
        x=x_edges, y=y_edges,
        mode="lines",
        line=dict(width=0.5, color="gray"),
        hoverinfo="none"
    )

    node_trace = go.Scatter(
        x=x_nodes, y=y_nodes,
        mode="markers",
        marker=dict(
            size=8,
            color=fiedler_values,
            colorscale="RdBu",
            colorbar=dict(title="Fiedler Value"),
            cmid=0.0,  # center colormap at zero
        ),
        text=hover_texts,
        hoverinfo="text"
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title,
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
    )

    fig.show()

def plot_spectral_embedding_2d(eigenvectors, x_index, y_index, node_group, reverse_map, title=None):
    """
    Plot a 2D spectral embedding with two eigenvectors as axes.
    Typically:
        x_index = 1 (Fiedler vector)
        y_index = 2, 3, 4... (any other eigenvector)
    """

    def normalize(v):
        vmax = np.abs(v).max()
        return v / vmax if vmax > 0 else v

    x = normalize(np.array(eigenvectors[x_index]))
    y = normalize(np.array(eigenvectors[y_index]))

    # Build hover text and node sizes
    hover_texts = []
    node_sizes  = []
    for n in range(len(x)):
        node_id   = reverse_map.get(n, "unknown")
        node_data = node_group.get(node_id, {})

        name  = node_data.get("name", "Unknown")
        stats = node_data.get("stats", [])

        if "isKeystone" in node_data:
            node_type = "Keystone"
            node_sizes.append(14)
        elif "isNotable" in node_data:
            node_type = "Notable"
            node_sizes.append(10)
        elif "isMastery" in node_data:
            node_type = "Mastery"
            node_sizes.append(7)
        elif "isJewelSocket" in node_data:
            node_type = "Jewel Socket"
            node_sizes.append(7)
        else:
            node_type = "Regular"
            node_sizes.append(5)

        stats_text = "<br>".join(stats) if stats else "No stats"
        hover_texts.append(
            f"<b>{name}</b><br>"
            f"Type: {node_type}<br>"
            f"ID: {node_id}<br>"
            f"λ{x_index}: {x[n]:.6f}<br>"
            f"λ{y_index}: {y[n]:.6f}<br>"
            f"Stats:<br>{stats_text}"
        )

    node_trace = go.Scatter(
        x=x.tolist(),
        y=y.tolist(),
        mode="markers",
        marker=dict(
            size=node_sizes,
            color=x.tolist(),           # color by x-axis eigenvector
            colorscale="Turbo",
            colorbar=dict(
                title=f"λ{x_index} (Fiedler)",
                tickfont=dict(color="white"),
            ),
            cmid=0.0,
            opacity=1.0,
            line=dict(
                width=0.5,
                color="white"
            ),
        ),
        text=hover_texts,
        hoverinfo="text"
    )

    auto_title = title or f"2D Spectral Embedding — λ{x_index} vs λ{y_index}"

    fig = go.Figure(
        data=[node_trace],
        layout=go.Layout(
            title=dict(
                text=auto_title,
                font=dict(color="white")
            ),
            hovermode="closest",
            paper_bgcolor="black",
            plot_bgcolor="rgb(15, 15, 25)",   # ← 2D uses plot_bgcolor instead of scene bgcolor
            xaxis=dict(
                title=f"Fiedler (λ{x_index})",
                tickfont=dict(color="white"),
                gridcolor="rgb(50, 50, 70)",
                zerolinecolor="white",
                zerolinewidth=1,
            ),
            yaxis=dict(
                title=f"λ{y_index}",
                tickfont=dict(color="white"),
                gridcolor="rgb(50, 50, 70)",
                zerolinecolor="white",
                zerolinewidth=1,
            ),
            margin=dict(l=0, r=0, b=40, t=40),
        )
    )

    fig.show()

def plot_spectral_embedding_3d(eigenvectors, x_index, y_index, z_index, node_group, reverse_map, title=None):

    def normalize(v):
        vmax = np.abs(v).max()
        return v / vmax if vmax > 0 else v

    x = normalize(np.array(eigenvectors[x_index]))
    y = normalize(np.array(eigenvectors[y_index]))
    z = normalize(np.array(eigenvectors[z_index]))

    # Build hover text and node sizes
    hover_texts = []
    node_sizes  = []
    for n in range(len(x)):
        node_id   = reverse_map.get(n, "unknown")
        node_data = node_group.get(node_id, {})

        name  = node_data.get("name", "Unknown")
        stats = node_data.get("stats", [])

        if "isKeystone" in node_data:
            node_type = "Keystone"
            node_sizes.append(14)
        elif "isNotable" in node_data:
            node_type = "Notable"
            node_sizes.append(10)
        elif "isMastery" in node_data:
            node_type = "Mastery"
            node_sizes.append(7)
        elif "isJewelSocket" in node_data:
            node_type = "Jewel Socket"
            node_sizes.append(7)
        else:
            node_type = "Regular"
            node_sizes.append(5)

        stats_text = "<br>".join(stats) if stats else "No stats"
        hover_texts.append(
            f"<b>{name}</b><br>"
            f"Type: {node_type}<br>"
            f"ID: {node_id}<br>"
            f"λ{x_index}: {x[n]:.6f}<br>"
            f"λ{y_index}: {y[n]:.6f}<br>"
            f"λ{z_index}: {z[n]:.6f}<br>"
            f"Stats:<br>{stats_text}"
        )

    node_trace = go.Scatter3d(
        x=x.tolist(),
        y=y.tolist(),
        z=z.tolist(),
        mode="markers",
        marker=dict(
            size=node_sizes,
            color=x.tolist(),
            colorscale="Turbo",     # ← much higher contrast than RdBu
            colorbar=dict(
                title=f"λ{x_index} (Fiedler)",
                tickfont=dict(color="white"),
            ),
            cmid=0.0,
            opacity=1.0,            # ← fully opaque
            line=dict(              # ← outline on each dot for extra contrast
                width=0.5,
                color="white"
            ),
        ),
        text=hover_texts,
        hoverinfo="text"
    )

    auto_title = title or f"3D Spectral Embedding — λ{x_index} vs λ{y_index} vs λ{z_index}"

    fig = go.Figure(
        data=[node_trace],
        layout=go.Layout(
            title=dict(
                text=auto_title,
                font=dict(color="white")  # ← white title text
            ),
            hovermode="closest",
            paper_bgcolor="black",        # ← black outer background
            scene=dict(
                bgcolor="rgb(15, 15, 25)",  # ← very dark navy for the 3D space
                xaxis=dict(
                    title=f"Fiedler (λ{x_index})",
                    tickfont=dict(color="white"),
                    gridcolor="rgb(50, 50, 70)",
                    zerolinecolor="white",
                ),
                yaxis=dict(
                    title=f"λ{y_index}",
                    tickfont=dict(color="white"),
                    gridcolor="rgb(50, 50, 70)",
                    zerolinecolor="white",
                ),
                zaxis=dict(
                    title=f"λ{z_index}",
                    tickfont=dict(color="white"),
                    gridcolor="rgb(50, 50, 70)",
                    zerolinecolor="white",
                ),
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )
    )

    fig.show()

def plot_spectral_clusters_2d(eigenvectors, cluster_labels, node_group, reverse_map, title="Spectral Clusters"):
    """
    2D cluster plot using Fiedler vector (x) and second eigenvector (y),
    with convex hull boundaries drawn around each cluster.
    """

    def normalize(v):
        vmax = np.abs(v).max()
        return v / vmax if vmax > 0 else v

    x = normalize(np.array(eigenvectors[1]))  # Fiedler
    y = normalize(np.array(eigenvectors[2]))  # second eigenvector
    labels = np.array(cluster_labels)
    n_clusters = len(set(labels))

    # Generate a distinct color for each cluster using a high contrast colorscale
    colorscale = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
        "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
        "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494",
    ]

    # Extend colors if more clusters than colors
    while len(colorscale) < n_clusters:
        colorscale += colorscale

    traces = []

    for cluster_id in sorted(set(labels)):
        mask        = labels == cluster_id
        cluster_x   = x[mask]
        cluster_y   = y[mask]
        color       = colorscale[cluster_id]
        node_indices = np.where(mask)[0]

        # Build hover text for this cluster's nodes
        hover_texts = []
        node_sizes  = []
        for n in node_indices:
            node_id   = reverse_map.get(int(n), "unknown")
            node_data = node_group.get(node_id, {})
            name      = node_data.get("name", "Unknown")
            stats     = node_data.get("stats", [])

            if "isKeystone" in node_data:
                node_type = "Keystone"
                node_sizes.append(14)
            elif "isNotable" in node_data:
                node_type = "Notable"
                node_sizes.append(10)
            elif "isJewelSocket" in node_data:
                node_type = "Jewel Socket"
                node_sizes.append(7)
            elif "classStartIndex" in node_data:
                node_type = "Class Start"
                node_sizes.append(30)
            else:
                node_type = "Regular"
                node_sizes.append(5)

            stats_text = "<br>".join(stats) if stats else "No stats"
            hover_texts.append(
                f"<b>{name}</b><br>"
                f"Type: {node_type}<br>"
                f"Cluster: {cluster_id}<br>"
                f"ID: {node_id}<br>"
                f"Stats:<br>{stats_text}"
            )

        # Draw convex hull boundary if cluster has enough points
        if len(cluster_x) >= 3:
            try:
                points = np.column_stack([cluster_x, cluster_y])
                hull   = ConvexHull(points)

                # Close the hull by appending the first point at the end
                hull_x = np.append(points[hull.vertices, 0], points[hull.vertices[0], 0])
                hull_y = np.append(points[hull.vertices, 1], points[hull.vertices[0], 1])

                traces.append(go.Scatter(
                    x=hull_x,
                    y=hull_y,
                    mode="lines",
                    line=dict(color=color, width=1.5, dash="dot"),
                    fill="toself",
                    fillcolor=color,
                    opacity=0.08,           # very faint fill
                    hoverinfo="none",
                    showlegend=False,
                ))
            except Exception:
                pass  # skip hull if points are collinear

        # Draw the nodes for this cluster
        traces.append(go.Scatter(
            x=cluster_x.tolist(),
            y=cluster_y.tolist(),
            mode="markers",
            name=f"Cluster {cluster_id}",
            marker=dict(
                size=node_sizes,
                color=color,
                line=dict(width=0.5, color="white"),
                opacity=1.0,
            ),
            text=hover_texts,
            hoverinfo="text",
        ))

    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=dict(text=title, font=dict(color="white")),
            hovermode="closest",
            paper_bgcolor="black",
            plot_bgcolor="rgb(15, 15, 25)",
            xaxis=dict(
                title="Fiedler Vector (λ₁)",
                tickfont=dict(color="white"),
                gridcolor="rgb(50, 50, 70)",
                zerolinecolor="white",
                zerolinewidth=1,
            ),
            yaxis=dict(
                title="Eigenvector 2 (λ₂)",
                tickfont=dict(color="white"),
                gridcolor="rgb(50, 50, 70)",
                zerolinecolor="white",
                zerolinewidth=1,
            ),
            legend=dict(
                font=dict(color="white"),
                bgcolor="rgba(0,0,0,0)",
            ),
            margin=dict(l=0, r=0, b=40, t=40),
        )
    )

    fig.show()

def plot_spectral_clusters_3d(eigenvectors, cluster_labels, node_group, reverse_map, title="Spectral Clusters 3D"):

    # Build the same embedding matrix KMeans used
    n_dims = min(40, len(eigenvectors) - 1)
    embedding = np.column_stack([
        eigenvectors[i] for i in range(1, n_dims + 1)
    ])

    # Normalize
    embedding = embedding / np.abs(embedding).max(axis=0)

    # ← 3D PCA instead of 2D
    pca = PCA(n_components=3)
    coords = pca.fit_transform(embedding)

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    labels     = np.array(cluster_labels)
    n_clusters = len(set(labels))

    colorscale = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
        "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
        "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494",
    ]
    while len(colorscale) < n_clusters:
        colorscale += colorscale

    traces = []

    for cluster_id in sorted(set(labels)):
        mask         = labels == cluster_id
        cluster_x    = x[mask]
        cluster_y    = y[mask]
        cluster_z    = z[mask]
        color        = colorscale[cluster_id]
        node_indices = np.where(mask)[0]

        # Build hover text and node sizes
        hover_texts = []
        node_sizes  = []
        for n in node_indices:
            node_id   = reverse_map.get(int(n), "unknown")
            node_data = node_group.get(node_id, {})
            name      = node_data.get("name", "Unknown")
            stats     = node_data.get("stats", [])

            if "isKeystone" in node_data:
                node_type = "Keystone"
                node_sizes.append(14)
            elif "isNotable" in node_data:
                node_type = "Notable"
                node_sizes.append(10)
            elif "isJewelSocket" in node_data:
                node_type = "Jewel Socket"
                node_sizes.append(7)
            elif "classStartIndex" in node_data:
                node_type = "Class Start"
                node_sizes.append(30)
            else:
                node_type = "Regular"
                node_sizes.append(5)

            stats_text = "<br>".join(stats) if stats else "No stats"
            hover_texts.append(
                f"<b>{name}</b><br>"
                f"Type: {node_type}<br>"
                f"Cluster: {cluster_id}<br>"
                f"ID: {node_id}<br>"
                f"Stats:<br>{stats_text}"
            )

        traces.append(go.Scatter3d(
            x=cluster_x.tolist(),
            y=cluster_y.tolist(),
            z=cluster_z.tolist(),
            mode="markers",
            name=f"Cluster {cluster_id}",
            marker=dict(
                size=node_sizes,
                color=color,
                line=dict(width=0.5, color="white"),
                opacity=1.0,
            ),
            text=hover_texts,
            hoverinfo="text",
        ))

    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=dict(text=title, font=dict(color="white")),
            hovermode="closest",
            paper_bgcolor="black",
            scene=dict(
                bgcolor="rgb(15, 15, 25)",
                xaxis=dict(
                    title="PCA 1",
                    tickfont=dict(color="white"),
                    gridcolor="rgb(50, 50, 70)",
                    zerolinecolor="white",
                ),
                yaxis=dict(
                    title="PCA 2",
                    tickfont=dict(color="white"),
                    gridcolor="rgb(50, 50, 70)",
                    zerolinecolor="white",
                ),
                zaxis=dict(
                    title="PCA 3",
                    tickfont=dict(color="white"),
                    gridcolor="rgb(50, 50, 70)",
                    zerolinecolor="white",
                ),
            ),
            legend=dict(
                font=dict(color="white"),
                bgcolor="rgba(0,0,0,0)",
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )
    )

    fig.show()