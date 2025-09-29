from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os

app = FastAPI(title="LMS Lightweight API")

class HealthCheckResponse(BaseModel):
    status: str
    version: str = "1.0.0"

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    return {"status": "healthy"}

# Add more lightweight endpoints as needed
# Heavy processing should be moved to separate serverless functions or external services
