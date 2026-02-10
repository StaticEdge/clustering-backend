from pathlib import Path
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
import matplotlib
matplotlib.use('Agg') # Required for non-interactive backend (Docker/Servers)
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from schemas.vulnerability import ClusteredVulnerability, ClusterSummary


class ClusteringService:
    def __init__(self):       # Using the higher-accuracy 12-layer model from the notebook
        self.model = SentenceTransformer('all-MiniLM-L12-v2')
        self.pca = PCA(n_components=30, random_state=42)
        self.latest_results = None
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        self.security_noise = {
            'vulnerability', 'cve', 'discovered', 'fixed', 'version',
            'attacker', 'exploit', 'security', 'issue', 'report',
            'product', 'software', 'allow', 'remote', 'local',
            'successfully', 'affected', 'described', 'information', 'patch'
        }
        self.stop_words = set(stopwords.words('english')).union(self.security_noise)
        self.latest_results = None

    def clean_label(self, label: str) -> str:
        if not isinstance(label, str): return "other"
        label = label.lower().replace('_', '-').strip()
        if label == 'cross-site-request-forgery-xss':
            label = 'cross-site-scripting-xss'
        return label

    def preprocess_text(self, text: str) -> str:
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        # Filter stopwords, security noise, and short tokens (> 2 chars)
        filtered = [w for w in words if w not in self.stop_words and len(w) > 2]
        return " ".join(filtered)

    def generate_visualization(self, df: pd.DataFrame, reduced_embeddings: np.ndarray):
        """Generates t-SNE plot and saves it to the static directory."""
        plt.figure(figsize=(12, 8))

        # t-SNE for 2D visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        vis_dims = tsne.fit_transform(reduced_embeddings)

        scatter = plt.scatter(
            vis_dims[:, 0],
            vis_dims[:, 1],
            c=df['cluster'],
            cmap='tab20',
            s=50,
            alpha=0.6
        )

        plt.title('Vulnerability Semantic Clusters', fontsize=15)
        plt.colorbar(scatter, label='Cluster ID')
        plt.grid(True, linestyle='--', alpha=0.5)

        # Save to static folder
        plt.savefig("cluster_plot.png")
        plt.close()
        return "/static/cluster_plot.png"

    async def run_clustering_pipeline(self, json_data: list):
        df = pd.DataFrame(json_data)
        if len(df) <= 1: return None

        # 1. Preprocessing & Label Cleaning
        df['clean_label'] = df['vulnerability_type'].apply(self.clean_label)
        df['processed_desc'] = df['description'].apply(self.preprocess_text)

        # 2. Vectorization & Dynamic PCA
        embeddings = self.model.encode(df['processed_desc'].tolist(), show_progress_bar=False)
        pca = PCA(n_components=min(30, len(df)), random_state=42)
        reduced_embeddings = pca.fit_transform(embeddings)

        # 3. Two-Step Agglomerative Clustering
        patch_count = len(df)
        clustering = AgglomerativeClustering(
            n_clusters=None, distance_threshold=0.85, metric='cosine', linkage='average'
        )
        labels = clustering.fit_predict(reduced_embeddings)

        # Enforce upper limit from notebook logic
        num_clusters = len(set(labels))
        max_clusters = (patch_count // 5) * 3 if patch_count > 9 else None

        if max_clusters and num_clusters > max_clusters:
            clustering = AgglomerativeClustering(
                n_clusters=max_clusters, metric='cosine', linkage='average'
            )
            labels = clustering.fit_predict(reduced_embeddings)

        df['cluster'] = labels

        # 4. Metrics & Mapping
        sil_score = None
        if 1 < len(set(labels)) < len(reduced_embeddings):
            sil_score = silhouette_score(reduced_embeddings, labels, metric='cosine')

        mapping = {}
        for cid in df['cluster'].unique():
            subset = df[df['cluster'] == cid]
            mapping[int(cid)] = Counter(subset['clean_label']).most_common(1)[0][0]

        df['inferred_label'] = df['cluster'].map(mapping)

        # 5. Serialization
        detailed = [ClusteredVulnerability(**row) for row in df.to_dict(orient='records')]
        summary = []
        for cid, label in mapping.items():
            cluster_subset = df[df['cluster'] == cid]
            summary.append(ClusterSummary(
                cluster_id=cid,
                predicted_type=label,
                record_count=len(cluster_subset),
                project_names=cluster_subset['name'].unique().tolist()
            ))

        # NEW: Trigger visualization generation
        plot_url = self.generate_visualization(df, reduced_embeddings)

        self.latest_results = {
            "status": "success",
            "plot_url": plot_url,  # URL to access the image
            "total_clusters": len(mapping),
            "global_silhouette_score": sil_score,
            "summary": summary,
            "detailed_results": detailed
        }
        return self.latest_results