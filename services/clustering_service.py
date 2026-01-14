import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from schemas.vulnerability import ClusteredVulnerability, ClusterSummary


class ClusteringService:
    def __init__(self):
        # Initialize the model once on startup
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.pca = PCA(n_components=50, random_state=42)
        self.latest_results = None

    def clean_text(self, text: str) -> str:
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'(/[a-zA-Z0-9._-]+)+', ' ', text)
        text = re.sub(r'\b[0-9a-fA-F]{7,40}\b', ' ', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    async def run_clustering_pipeline(self, json_data: list):
        df = pd.DataFrame(json_data)

        # 1. Preprocessing
        df['clean_label'] = df['vulnerabilitytype'].str.lower().str.replace('_', '-').strip()
        df['processed_desc'] = df['description'].apply(self.clean_text)

        # 2. Vectorization & Reduction
        embeddings = self.model.encode(df['processed_desc'].tolist())
        reduced_embeddings = self.pca.fit_transform(embeddings)

        # 3. Clustering
        clustering = AgglomerativeClustering(
            n_clusters=None, distance_threshold=0.55, metric='cosine', linkage='average'
        )
        df['cluster'] = clustering.fit_predict(reduced_embeddings)

        # 4. Map Clusters to Majority Label
        mapping = {}
        for cid in df['cluster'].unique():
            subset = df[df['cluster'] == cid]
            mapping[int(cid)] = Counter(subset['clean_label']).most_common(1)[0][0]

        df['inferred_label'] = df['cluster'].map(mapping)

        # 5. Transform to Pydantic Objects
        detailed = [ClusteredVulnerability(**row) for row in df.to_dict(orient='records')]

        summary = []
        for cid, label in mapping.items():
            projects = df[df['cluster'] == cid]['name'].unique().tolist()
            summary.append(ClusterSummary(
                cluster_id=cid,
                predicted_vulnerability_type=label,
                project_names=projects
            ))

        self.latest_results = {
            "status": "success",
            "total_clusters": len(mapping),
            "summary": summary,
            "detailed_results": detailed
        }
        return self.latest_results