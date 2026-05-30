#This Module holds the models for the DB
from sqlalchemy import Column, Integer, String, Float, JSON
from db.database import Base

class SkillTreeResult(Base):
    __tablename__ = "skill_tree_results"

    id = Column(Integer, primary_key=True, index=True)
    group_key = Column(String, index=True)
    node_count = Column(Integer)
    special_node_count = Column(Integer)
    edges = Column(JSON) #stores edges as JSON array of tuples
    eigenvalues = Column(JSON) #stores Vec<f64> as JSON
    eigenvectors = Column(JSON) #stores Vec<Vec<f64>> as nested JSON

class ClusteringResult(Base):
    __tablename__ = "clustering_results"

    id = Column(Integer, primary_key=True, index=True)
    group_key = Column(String, index=True)
    clusters = Column(JSON) #stores Vec<usize> as JSON

class ClusterLabel(Base):
    __tablename__ = "cluster_labels"

    id = Column(Integer, primary_key=True, index=True)
    group_key = Column(String, index=True)
    cluster_id = Column(Integer) # Matches KMeans cluster Index
    tags = Column(JSON) # Stores Vec<String> as JSON, i.e. ["physical", "attack", "damage"]
    top_stats = Column(JSON) # raw top N stats for reference