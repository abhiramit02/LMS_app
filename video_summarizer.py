from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pytube import YouTube
import os

# Prevent Transformers from importing TensorFlow/Flax to avoid Keras 3 incompatibility
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

router = APIRouter()

# Lazy-load the Hugging Face summarizer to avoid heavy imports at module load
_summarizer = None

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        # Import here so env flags above take effect and to delay heavy initialization
        from transformers import pipeline
        # Use a lighter, PyTorch-based model and force PyTorch framework
        _summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            framework="pt",
        )
    return _summarizer

# Input schema for request
class YouTubeURL(BaseModel):
    url: str

@router.post("/summarize-youtube")
async def summarize_youtube(request: YouTubeURL):
    try:
        yt = YouTube(request.url)

        # Extract video title and description (not downloading full video for now)
        title = yt.title
        description = yt.description

        if not description.strip():
            raise HTTPException(status_code=400, detail="No description found to summarize.")

        # Summarize description text
        summary = get_summarizer()(description, max_length=120, min_length=40, do_sample=False)

        return {
            "video_title": title,
            "summary": summary[0]["summary_text"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
