import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from schemas.vulnerability import ClusteredVulnerability, ClusterSummary


class ClusteringService:
    def __init__(self):
        # Using the L12 model as per notebook requirements
        self.model = SentenceTransformer('all-MiniLM-L12-v2')
        self.pca_components = 25  # Updated from notebook best params
        self.dist_threshold = 0.65  # Updated from notebook best params

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
        if 'xss' in label: return 'cross-site-scripting-xss'
        return label

    def preprocess_text(self, text: str) -> str:
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        # Filter noise and short tokens (> 2 chars) matching notebook logic
        filtered = [w for w in words if w not in self.stop_words and len(w) > 2]
        return " ".join(filtered)

    def generate_visualization(self, reduced_embeddings: np.ndarray, labels: np.ndarray):
        """Generates a t-SNE plot for the clusters."""
        plt.figure(figsize=(10, 7))
        # Use t-SNE to project the 25 PCA components down to 2D for plotting
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(reduced_embeddings) - 1))
        vis_dims = tsne.fit_transform(reduced_embeddings)

        scatter = plt.scatter(vis_dims[:, 0], vis_dims[:, 1], c=labels, cmap='tab20', s=50, alpha=0.7)
        plt.colorbar(scatter, label='Cluster ID')
        plt.title(f'Cluster Visualization (Threshold: 0.65, PCA: 25)')
        plt.savefig("cluster_plot.png")
        plt.close()
        return "/visualization"

    async def run_clustering_pipeline(self, json_data: list):
        df = pd.DataFrame(json_data)
        if len(df) < 2: return None

        # 1. Preprocessing
        df['clean_label'] = df['vulnerability_type'].apply(self.clean_label)
        df['processed_desc'] = df['description'].apply(self.preprocess_text)

        # 2. Vectorization & PCA (Fixed 25 components)
        embeddings = self.model.encode(df['processed_desc'].tolist(), show_progress_bar=False)

        n_comp = min(self.pca_components, len(df))
        pca = PCA(n_components=n_comp, random_state=42)
        reduced_embeddings = pca.fit_transform(embeddings)

        # 3. Agglomerative Clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.dist_threshold,
            metric='cosine',
            linkage='average'
        )
        labels = clustering.fit_predict(reduced_embeddings)
        df['cluster'] = labels

        # 4. Generate Visualization
        plot_url = self.generate_visualization(reduced_embeddings, labels)

        # 5. Metrics & Labeling
        sil_score = None
        if 1 < len(set(labels)) < len(reduced_embeddings):
            sil_score = float(silhouette_score(reduced_embeddings, labels, metric='cosine'))

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

        self.latest_results = {
            "status": "success",
            "total_clusters": len(mapping),
            "global_silhouette_score": sil_score,
            "summary": summary,
            "detailed_results": detailed
        }
        return self.latest_results