from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from schemas.vulnerability import ClusteringResponse
from services.clustering_service import ClusteringService
import json
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import asyncio

# ---------------- APP & SERVICE ----------------

# Initialize the service (this will trigger the S-BERT model download/load)
cluster_service = ClusteringService()
observer: Observer | None = None

# ---------------- PATH CONFIG ----------------

# Resolves to the directory containing main.py (Crucial for Docker)
JSON_FILE = Path("gemini_vulnerability_signatures_2022.json")


# ---------------- FILE WATCHER ----------------

class PatchFileHandler(FileSystemEventHandler):
    """
    Monitors the 'data' directory for changes to patch_analysis.json.
    """

    def on_modified(self, event):
        if event.src_path.endswith("gemini_vulnerability_signatures_2022.json"):
            print(f"Change detected in {JSON_FILE}. Re-clustering...")
            asyncio.create_task(trigger_recluster())

async def trigger_recluster():
    if JSON_FILE.exists():
        with open(JSON_FILE, "r") as f:
            await cluster_service.run_clustering_pipeline(json.load(f))
        print("Clusters updated from local JSON.")

async def lifespan(app: FastAPI):
    global observer
    await trigger_recluster()
    observer = Observer()
    observer.schedule(PatchFileHandler(), path=".", recursive=False)
    observer.start()
    yield
    observer.stop()

# ---------------- FASTAPI APP ----------------

app = FastAPI(
    title="Semgrep Rule Gen Backend",
    description="Clusters vulnerability descriptions using semantic embeddings and dynamic Agglomerative Clustering.",
    lifespan=lifespan
)
print("🔥 LOADED NEW MAIN.PY 🔥")


# ---------------- ENDPOINTS ----------------

@app.post("/cluster")
async def cluster_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".json"):
        raise HTTPException(400, "Only JSON files allowed")

    data = json.load(file.file)
    await cluster_service.run_clustering_pipeline(data)

    return {
        "status": "clustering completed",
        "records": len(data)
    }

@app.get("/results", response_model=ClusteringResponse)
async def get_results():
    if not cluster_service.latest_results:
        raise HTTPException(400, "Run clustering first")
    return cluster_service.latest_results

@app.get("/visualization")
async def get_visualization():
    plot = Path("cluster_plot.png")
    if not plot.exists():
        raise HTTPException(400, "Visualization not generated yet")
    return FileResponse(plot)