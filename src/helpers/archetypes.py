import re
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Reference for words that appear commonly but carry no meaning for us
STOPWORDS = {
    "increased", "decreased", "more", "less", "to", "of", "and",
    "with", "for", "you", "your", "by", "per", "the", "a", "an",
    "at", "in", "is", "are", "has", "have", "been", "be", "or",
    "that", "this", "when", "while", "if", "up", "from", "as",
    "not", "no", "can", "cannot", "each", "all", "also", "any",
    "recently", "nearby", "maximum", "minimum", "additional",
    "base", "over", "gain", "grants", "deal", "deals", "take",
    "takes", "hits", "hit", "against", "affected", "applies",
    "applied", "only", "once", "chance", "seconds", "second",
    "effect", "effects", "bonus", "value", "rate", "level",
    "skills", "skill", "action", "speed", "duration", "least", "one",
    "corpse"
}

def clean_stat(stat: str) -> list[str]:
    #remove numbers and symbols
    stat = re.sub(r"[\d\.\+\-\%]", " ", stat.lower())

    #remove punctuation
    stat = re.sub(r"[^a-z\s]", " ", stat)

    #split and filter
    words = [
        word for word in stat.split()
        if len(word) > 2 and word not in STOPWORDS
    ]
    print(f"Cleaned stat: {words}")
    return words

def build_cluster_corpus(cluster_labels, node_group, reverse_map):
    """
    Build a text corpus where each document is all the stats
    from a given cluster concatenated together.
    returns a dict of {cluster_id: [list of all words]}
    """
    cluster_words = {}

    for node_idx, cluster_id in enumerate(cluster_labels):
        node_id = reverse_map.get(node_idx, "unknown")
        node_data = node_group.get(node_id, {})
        stats = node_data.get("stats", [])

        if cluster_id not in cluster_words:
            cluster_words[cluster_id] = []
        
        for stat in stats:
            cluster_words[cluster_id].extend(clean_stat(stat))

    return cluster_words

def frequency_analysis(cluster_words, top_n=20):
    """
    Run word frequency analysis per cluster
    Returns a dict of {cluster_id: [(word, frequency),...]}
    """

    cluster_freq = {}

    for cluster_id, words in cluster_words.items():
        counter = Counter(words)
        cluster_freq[cluster_id] = counter.most_common(top_n)
    return cluster_freq

def generate_tags_tfidf(cluster_words, top_n_tags = 5):
    """
    Use TF-IDF to find the most distinctive words per cluster.
    TF-IDF penalizes words that appear in many clusters (like 'damage')
    and rewards words that are unique to a specific cluster.
    Returns {cluster_id: [tag1, tag2, ...]}
    """
    cluster_ids = sorted(cluster_words.keys())

    #build one document per cluster
    documents = [
        " ".join(cluster_words[cid]) for cid in cluster_ids
    ]

    #fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features = 500,
        ngram_range = (1, 2),
        min_df = 1
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    #Extract top N tags per cluster
    cluster_tags = {}
    for idx, cluster_id in enumerate(cluster_ids):
        row = tfidf_matrix[idx].toarray()[0]
        top_indicies = row.argsort()[-top_n_tags:][::-1]
        tags = [feature_names[i] for i in top_indicies if row[i] > 0]
        cluster_tags[cluster_id] = tags

    return cluster_tags

def match_prompt_to_clusters(prompt, cluster_tags, top_n=3):
    """
    Given a build prompt, find the top N most relevant clusters
    using TF-IDF cosine similarity.
    Returns list of (cluster_id, score) sorted by relevance.
    """
    cluster_ids = sorted(cluster_tags.keys())

    #rebuild corpus from tags for vectorization
    documents = [
        " ".join(cluster_tags[cid]) for cid in cluster_ids
    ]
    documents.append(prompt.lower()) #prompt is the last document

    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    tfidf_matrix = vectorizer.fit_transform(documents)

    #Cosine similarity between prompt and each cluster
    prompt_vector = tfidf_matrix[-1]
    cluster_vecs = tfidf_matrix[:-1]
    similarities = cosine_similarity(prompt_vector, cluster_vecs)[0]

    #sort by similarity
    ranked = sorted(
        zip(cluster_ids, similarities),
        key= lambda x: x[1],
        reverse=True
    )

    return ranked[:top_n]