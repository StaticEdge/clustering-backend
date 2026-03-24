from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from schemas.vulnerability import ClusteringResponse
from services.clustering_service import ClusteringService
import json
from pathlib import Path
import uvicorn

cluster_service = ClusteringService()
JSON_FILE = Path("gemini_vulnerability_signatures_2022.json")

async def lifespan(app: FastAPI):
    # Initial load if file exists
    if JSON_FILE.exists():
        with open(JSON_FILE, "r") as f:
            await cluster_service.run_clustering_pipeline(json.load(f))
    yield

app = FastAPI(title="Patch Clustering Backend", lifespan=lifespan)

@app.post("/cluster")
async def cluster_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".json"):
        raise HTTPException(400, "Only JSON files allowed")

    try:
        content = await file.read()
        data = json.loads(content)
        result = await cluster_service.run_clustering_pipeline(data)
        return {"status": "success", "clusters_found": result["total_clusters"]}
    except Exception as e:
        raise HTTPException(500, f"Error processing file: {str(e)}")

@app.get("/results", response_model=ClusteringResponse)
async def get_results():
    if not cluster_service.latest_results:
        raise HTTPException(404, "No clustering results available. Please upload a file first.")
    return cluster_service.latest_results

@app.get("/visualization")
async def get_visualization():
    plot_path = Path("cluster_plot.png")
    if not plot_path.exists():
        raise HTTPException(404, "Plot not found")
    return FileResponse(plot_path)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)