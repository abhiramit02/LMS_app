from fastapi import FastAPI, Request
import pandas as pd
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os

from video_summarizer import router as lecture_router
from routes.assignments import router as assignment_router
from assignment_generator import router as file_assignment_router
from routes.recommendations import router as recommendation_router
from routes.plagiarism import router as plagiarism_router
from routes.analytics import router as analytics_router
from routes.pdf_chatbot import router as pdf_chatbot_router
from routes.review_summarizer import router as review_summarizer_router
from routes.student_progress import router as student_progress_router

# Allow disabling grading microservice (heavy deps) for serverless deploys
DISABLE_GRADING = os.getenv("DISABLE_GRADING", "0") == "1"
if not DISABLE_GRADING:
    from grading import app as grading_app  # Mount the grading service under /grading
else:
    grading_app = None

app = FastAPI(title="LMS Web Service")
try:
    from recommendation_system import SomeClass
except ImportError as e:
    print(f"Warning: recommendation_system import failed: {e}")

# API routers
app.include_router(assignment_router, prefix="/assignments", tags=["assignments"])
app.include_router(lecture_router, prefix="/lecture", tags=["lecture"])
app.include_router(file_assignment_router, prefix="/file", tags=["file-based-assignments"])
app.include_router(recommendation_router, prefix="/recommendations", tags=["recommendations"])
app.include_router(plagiarism_router, prefix="/plagiarism", tags=["plagiarism"])
app.include_router(analytics_router, prefix="/analytics", tags=["analytics"])
app.include_router(pdf_chatbot_router, prefix="/pdf-chatbot", tags=["pdf-chatbot"])
app.include_router(review_summarizer_router, prefix="/review-summarizer", tags=["review-summarizer"])
app.include_router(student_progress_router, prefix="/student-progress", tags=["student-progress"])

# Mount grading microservice ASGI app (if enabled)
if grading_app is not None:
    app.mount("/grading", grading_app)

# Templates and static assets
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="templates"), name="static")

# Web UI routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "title": "LMS Dashboard"})

@app.get("/ui/assignments", response_class=HTMLResponse)
async def ui_assignments(request: Request):
    return templates.TemplateResponse("assignments.html", {"request": request, "title": "Assignment Generator"})

@app.get("/ui/file-assignment", response_class=HTMLResponse)
async def ui_file_assignment(request: Request):
    return templates.TemplateResponse("file_assignment.html", {"request": request, "title": "File-based Assignment Generator"})

@app.get("/ui/grading", response_class=HTMLResponse)
async def ui_grading(request: Request):
    return templates.TemplateResponse("grading.html", {"request": request, "title": "Automated Grading"})

@app.get("/ui/recommendations", response_class=HTMLResponse)
async def ui_recommendations(request: Request):
    return templates.TemplateResponse("recommendations.html", {"request": request, "title": "AI Learning Recommendations"})

@app.get("/ui/plagiarism", response_class=HTMLResponse)
async def ui_plagiarism(request: Request):
    return templates.TemplateResponse("plagiarism.html", {"request": request, "title": "Plagiarism Checker"})

@app.get("/ui/analytics", response_class=HTMLResponse)
async def ui_analytics(request: Request):
    return templates.TemplateResponse("analytics.html", {"request": request, "title": "Performance Analytics"})

@app.get("/ui/pdf-chatbot", response_class=HTMLResponse)
async def ui_pdf_chatbot(request: Request):
    return templates.TemplateResponse("pdf_chatbot.html", {"request": request, "title": "PDF Teaching Assistant"})

@app.get("/ui/review-summarizer", response_class=HTMLResponse)
async def ui_review_summarizer(request: Request):
    return templates.TemplateResponse("review_summarizer.html", {"request": request, "title": "Teacher Review Summarizer"})

@app.get("/ui/student-progress", response_class=HTMLResponse)
async def ui_student_progress(request: Request):
    return templates.TemplateResponse("student_progress.html", {"request": request, "title": "Student Progress Tracker"})

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/students")
def get_students():
    df = pd.read_csv("data/student_activity.csv")
    return df.to_dict(orient="records")
