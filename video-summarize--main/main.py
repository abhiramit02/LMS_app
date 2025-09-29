import os
import nltk

# Make sure required NLTK resources are available
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

import re
import uuid
import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from video_pipeline import (
    extract_audio,
    transcribe_audio,
    save_transcript,
    abstractive_summary,
    extractive_summary,
    generate_structured_notes
)

from yt_dlp import YoutubeDL

# ─── FastAPI Setup ────────────────────────────────────────────────────────────

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def root():
    return FileResponse("frontend/index.html")

# Ensure uploads directory exists
os.makedirs("uploads", exist_ok=True)


# ─── Utility: Extract YouTube Video ID ────────────────────────────────────────

def extract_video_id(url: str) -> str:
    """
    Pull the 11‑char YouTube ID from any valid URL form.
    Raises ValueError if it can't find it.
    """
    pattern = re.compile(
        r"""(?:v=|\/shorts\/|youtu\.be\/)
            ([0-9A-Za-z_-]{11})
            (?=$|[?&])""",
        re.VERBOSE
    )
    match = pattern.search(url)
    if not match:
        raise ValueError(f"Could not extract a YouTube video ID from `{url}`")
    return match.group(1)


# ─── Endpoint: Upload Video File ─────────────────────────────────────────────

@app.post("/process-youtube")
async def process_youtube_url(url: str = Form(...)):
    try:
        print("Incoming URL:", url)
        if not url.strip():
            raise HTTPException(status_code=400, detail="No YouTube URL provided.")
        # 1) Extract canonical video ID
        try:
            video_id = extract_video_id(url)
        except ValueError as ve:
            print("ID extraction failed:", ve)
            raise HTTPException(status_code=400, detail=f"Invalid YouTube URL: {ve}")

        # 2) Normalize to standard watch?v= URL
        clean_url = f"https://www.youtube.com/watch?v={video_id}"
        print("Normalized URL →", clean_url)

        # 3) Download audio using yt_dlp
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'uploads/yt_{video_id}.%(ext)s',
            'quiet': True,
            'noplaylist': True
        }

        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(clean_url, download=True)
                download_path = ydl.prepare_filename(info)
                print("Downloaded to:", download_path)
        except Exception as e:
            print("yt-dlp error:", e)
            raise HTTPException(status_code=400, detail=f"Failed to download YouTube audio: {e}")

        # 4) Process it
        return await process_media(download_path)

    except HTTPException:
        raise
    except Exception as e:
        print("Unexpected error in /process-youtube:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")



# ─── Core Media Processing ──────────────────────────────────────────────────

async def process_media(media_path: str):
    try:
        if not os.path.exists(media_path):
            raise FileNotFoundError(f"Media file not found: {media_path}")

        print("Processing media from:", media_path)

        # 1) Extract audio
        audio_path = extract_audio(media_path)
        print("Audio extracted at:", audio_path)

        # 2) Transcribe
        transcript = transcribe_audio(audio_path)
        print("Transcript:", transcript)

        # 3) Save transcript
        save_transcript(transcript)
        print("Transcript saved.")

        # 4) Summaries
        summary_abs = abstractive_summary(transcript)
        print("Abstractive Summary:", summary_abs)

        summary_ext = extractive_summary(transcript)
        print("Extractive Summary:", summary_ext)

        # 5) Structured notes
        structured_notes = generate_structured_notes(summary_abs)
        print("Structured Notes Generated:", structured_notes)

        # 6) Cleanup
        os.remove(media_path)
        os.remove(audio_path)
        print("Removed temporary files.")

        # 7) Return JSON
        return JSONResponse({
            "transcript": transcript,
            "summary_abstractive": summary_abs,
            "summary_extractive": summary_ext,
            "structured_notes": structured_notes
        })

    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except Exception as e:
        print("Error in process_media:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing media: {e}")


# ─── Endpoint: Upload Video File ─────────────────────────────────────────────

@app.post("/process-video")
async def process_uploaded_video(video: UploadFile = File(...)):
    try:
        filename = f"uploads/{uuid.uuid4()}_{video.filename}"
        with open(filename, "wb") as f:
            f.write(await video.read())

        return await process_media(filename)

    except Exception as e:
        print("Error in /process-video:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing uploaded video: {e}")
