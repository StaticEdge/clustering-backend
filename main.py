from fastapi import FastAPI, BackgroundTasks, HTTPException
from schemas.vulnerability import ClusteringResponse, ClusterSummary
from services.clustering_service import ClusteringService
from typing import List
import json
import os
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import asyncio

app = FastAPI(title="Semgrep Rule Gen Backend")
cluster_service = ClusteringService()

# 1. Use Absolute Paths for Reliability
BASE_DIR = Path(__file__).resolve().parent
JSON_FILE_PATH = BASE_DIR / "data" / "patch_analysis.json"

# --- AUTOMATED FILE WATCHER LOGIC ---
class PatchFileHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path == str(JSON_FILE_PATH):
            print(f"Detected change in {JSON_FILE_PATH}. Re-clustering...")
            self.trigger_recluster()

    def trigger_recluster(self):
        with open(JSON_FILE_PATH, 'r') as f:
            data = json.load(f)
        # We use a thread-safe way to run the async pipeline
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(cluster_service.run_clustering_pipeline(data))
        print("Auto-update complete.")

def start_file_watcher():
    observer = Observer()
    handler = PatchFileHandler()
    # Watch the directory containing the file
    observer.schedule(handler, path=str(JSON_FILE_PATH.parent), recursive=False)
    observer.start()

@app.on_event("startup")
async def startup_event():
    # Start the watcher when the API starts
    if JSON_FILE_PATH.exists():
        print("Starting file watcher...")
        start_file_watcher()
        # Run initial clustering on startup
        with open(JSON_FILE_PATH, 'r') as f:
            data = json.load(f)
        await cluster_service.run_clustering_pipeline(data)
    else:
        print(f"WARNING: Initial file not found at {JSON_FILE_PATH}")

# --- ENDPOINTS ---

@app.post("/trigger-update", status_code=202)
async def update_clusters(background_tasks: BackgroundTasks):
    if not JSON_FILE_PATH.exists():
        raise HTTPException(status_code=404, detail=f"File not found at {JSON_FILE_PATH}")

    async def run_pipeline():
        with open(JSON_FILE_PATH, 'r') as f:
            data = json.load(f)
        await cluster_service.run_clustering_pipeline(data)

    background_tasks.add_task(run_pipeline)
    return {"message": "Clustering update initiated manually."}

@app.get("/results", response_model=ClusteringResponse)
async def get_full_results():
    if not cluster_service.latest_results:
        raise HTTPException(status_code=400, detail="No clustering results available.")
    return cluster_service.latest_results