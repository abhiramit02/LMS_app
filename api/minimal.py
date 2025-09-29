from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
import os

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="templates/static"), name="static")
templates = Jinja2Templates(directory="templates")

# Serve index.html for all routes
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Add catch-all route for all other paths
@app.get("/{full_path:path}")
async def catch_all(request: Request, full_path: str):
    # If the path starts with ui/, serve index.html (client-side routing)
    if full_path.startswith("ui/"):
        return templates.TemplateResponse("index.html", {"request": request})
    # Otherwise, return 404
    return {"detail": "Not Found"}

# Add specific API routes if needed
@app.get("/api/health")
async def health_check():
    return {"status": "ok"}