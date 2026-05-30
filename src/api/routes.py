# This Module holds the routes for the backend api
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from db.database import SessionLocal, engine, Base
from api.models import SkillTreeResult, ClusteringResult
import pyo3_example
from helpers.graph_helpers import build_edges, build_grouped_nodes, detect_branches

import json
from pathlib import Path

Base.metadata.create_all(bind=engine) #creates tables if they don't exist

app = FastAPI()

# Load and group nodes once at startup so all routes can access them
base_path = Path(__file__).parent.parent  # points to src/
file_path = base_path / "../assets/data.json"
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

grouped_nodes = build_grouped_nodes(data)  # ← built once at startup, reused by all routes

#opens and closes DB session per request
def get_db():
    db = SessionLocal()
    try: 
        yield db
    finally:
        db.close()

@app.post("/compute/eigenvectors/{group_key}")
def compute_tree(group_key: str, db: Session = Depends(get_db)):

    # Check if already computed
    existing = db.query(SkillTreeResult).filter(
        SkillTreeResult.group_key == group_key
    ).first()
    if existing:
        return {"message": f"{group_key} already computed, skipping"}

    node_group = grouped_nodes[group_key]
    edges, node_count, node_weights, reverse_map = build_edges(node_group)
    eigenvectors, eigenvalues = pyo3_example.compute_laplacian(edges, node_weights)

    # Count special nodes
    special_node_count = sum(
        1 for node_data in node_group.values()
        if "isKeystone" in node_data or "isNotable" in node_data
    )

    result = SkillTreeResult(
        group_key=group_key,
        node_count=node_count,
        special_node_count=special_node_count,
        edges=edges,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors
    )
    db.add(result)
    db.commit()

    return {
        "message":       f"Computed and stored {group_key}",
        "node_count":    node_count,
        "special_node_count": special_node_count
    }

@app.post("/compute/clustering/{group_key}")
def compute_clustering(group_key: str, db: Session = Depends(get_db)):

    existing = db.query(ClusteringResult).filter(
        ClusteringResult.group_key == group_key
    ).first()

    if existing:
        return {"message": f"{group_key} already computed, skipping"}

    results = db.query(SkillTreeResult).filter(
        SkillTreeResult.group_key == group_key
    ).first()
    if not results:
        return {"error": f"No decomposition results found for {group_key}"}

    print(f"Special node count:")
    
    eigenvectors = results.eigenvectors
    k = results.special_node_count - 1
    if len(eigenvectors) <= 50:
        print(f"Small graph({len(eigenvectors)} nodes), using branch detection")
        clusters = detect_branches(results.edges, results.node_count)
    else:
        print(f"Large graph({len(eigenvectors)} nodes), using spectral clustering")
        clusters = pyo3_example.run_spectral_kmeans(eigenvectors, 7)

    clustering_result = ClusteringResult(
        group_key=group_key,
        clusters=clusters
    )

    db.add(clustering_result)
    db.commit()

    return {
        "message": f"Computed and stored clustering for {group_key}",
        "clusters": clusters
    }


@app.get("/results/eigenvectors/{group_key}")
def get_results(group_key: str, db: Session = Depends(get_db)):
    """Retrieve stored edges and eigenvectors for a skill tree group."""
    result = db.query(SkillTreeResult).filter(
        SkillTreeResult.group_key == group_key
    ).first()

    if not result:
        return {"error": f"No results found for {group_key} — try POST /compute/{group_key} first"}

    return {
        "group_key":   result.group_key,
        "node_count":  result.node_count,
        "special_node_count": result.special_node_count,
        "edges":       result.edges,
        "eigenvectors": result.eigenvectors,
    }

@app.get("/results/clustering/{group_key}")
def get_clustering_results(group_key: str, db: Session = Depends(get_db)):
    results = db.query(ClusteringResult).filter(
        ClusteringResult.group_key == group_key
    ).first()
    if not results:
        return {"error": f"Not all data need could be found for {group_key}"}
    
    return {
        "clusters": results.clusters
    }

@app.delete("/results/eigenvectors/{group_key}")
def delete_results(group_key: str, db: Session = Depends(get_db)):
    """Delete stored results so they can be recomputed fresh."""
    result = db.query(SkillTreeResult).filter(
        SkillTreeResult.group_key == group_key
    ).first()

    if not result:
        return {"error": f"No results found for {group_key}"}

    db.delete(result)
    db.commit()
    return {"message": f"Deleted results for {group_key}"}

@app.delete("/results/clustering/{group_key}")
def delete_clustering_results(group_key: str, db: Session = Depends(get_db)):
    results = db.query(ClusteringResult).filter(
        ClusteringResult.group_key == group_key
    ).first()

    if not results:
        return {"error": f"No clustering results found for {group_key}"}

    db.delete(results)
    db.commit()
    return {"message": f"Deleted clustering results for {group_key}"}

@app.get("/testing")
def testing(db: Session = Depends(get_db)):
    from helpers.archetypes import build_cluster_corpus, frequency_analysis, generate_tags_tfidf, match_prompt_to_clusters

    results = db.query(ClusteringResult).filter(
        ClusteringResult.group_key == "main"
    ).first()
    if not results:
        return {"error": f"No clustering results found for main"}
    
    clusters = results.clusters
    node_group = grouped_nodes["main"]
    _, _, _, reverse_map = build_edges(node_group)

    cluster_words = build_cluster_corpus(clusters, node_group, reverse_map)
    cluster_freq = frequency_analysis(cluster_words)
    print(f"Cluster frequency: {cluster_freq}")
    cluster_tags = generate_tags_tfidf(cluster_words)
    print(f"Cluster tags: {cluster_tags}")

    prompt = "Create a build for me that uses wands and minions for high damage and energy shields for defense"
    ranked = match_prompt_to_clusters(prompt, cluster_tags, top_n=5)
    print(f"Prompt matched to clusters: {ranked}")
    
    return {"cluster_words": cluster_words, "cluster_freq": cluster_freq, "cluster_tags": cluster_tags, "ranked": ranked}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)