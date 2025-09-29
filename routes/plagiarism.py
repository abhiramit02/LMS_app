from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
from pathlib import Path
import tempfile
import shutil

from plagiarism_checker import PlagiarismChecker

router = APIRouter(tags=["plagiarism"])

# Initialize plagiarism checker
plagiarism_checker = PlagiarismChecker()

class PlagiarismCheckRequest(BaseModel):
    student_id: str
    assignment_title: str
    file_path: Optional[str] = None

class PlagiarismResponse(BaseModel):
    plagiarism_detected: bool
    similarity_score: float
    plagiarism_percentage: float
    severity: str
    alert_message: str
    detailed_comparisons: List[Dict]
    timestamp: str

@router.post("/check", response_model=PlagiarismResponse)
async def check_plagiarism(
    file: UploadFile = File(...),
    student_id: str = Form(...),
    assignment_title: str = Form(...)
):
    """
    Check uploaded assignment for plagiarism
    """
    try:
        # Validate file type
        allowed_extensions = ['.pdf', '.docx', '.txt']
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        try:
            # Perform plagiarism check
            result = plagiarism_checker.detect_plagiarism(
                temp_file_path, 
                student_id, 
                assignment_title
            )
            
            # Extract text content for future comparison
            if 'error' not in result:
                try:
                    content = plagiarism_checker.extract_text_from_file(temp_file_path)
                    plagiarism_checker.save_assignment_for_comparison(
                        student_id, 
                        assignment_title, 
                        content, 
                        temp_file_path
                    )
                except Exception as e:
                    print(f"Warning: Failed to save assignment for comparison: {e}")
            
            # Ensure the result matches the expected response model
            if 'error' in result:
                raise HTTPException(status_code=400, detail=result['error'])
            
            response_data = {
                "plagiarism_detected": result.get('plagiarism_detected', False),
                "similarity_score": result.get('similarity_score', 0.0),
                "plagiarism_percentage": result.get('plagiarism_percentage', 0.0),
                "severity": result.get('severity', 'NONE'),
                "alert_message": result.get('alert_message', 'No message'),
                "detailed_comparisons": result.get('detailed_comparisons', []),
                "timestamp": result.get('timestamp', '')
            }
            
            return response_data
            
        finally:
            try:
                os.unlink(temp_file_path)
            except:
                pass
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plagiarism check failed: {str(e)}")

@router.get("/statistics")
async def get_plagiarism_statistics():
    try:
        return plagiarism_checker.get_plagiarism_statistics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@router.post("/save-assignment")
async def save_assignment_for_comparison(
    student_id: str = Form(...),
    assignment_title: str = Form(...),
    content: str = Form(...),
    file_path: Optional[str] = Form(None)
):
    try:
        success = plagiarism_checker.save_assignment_for_comparison(
            student_id, assignment_title, content, file_path
        )
        if success:
            return {"message": "Assignment saved successfully", "status": "success"}
        else:
            raise HTTPException(status_code=500, detail="Failed to save assignment")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save assignment: {str(e)}")

@router.get("/export-report")
async def export_plagiarism_report():
    try:
        output_path = plagiarism_checker.export_plagiarism_report()
        if output_path:
            return {"message": "Report exported successfully", "file_path": output_path}
        else:
            raise HTTPException(status_code=500, detail="Failed to export report")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export report: {str(e)}")

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "plagiarism-detection",
        "models_loaded": {
            "tfidf": plagiarism_checker.tfidf_vectorizer is not None,
            "sentence_transformer": plagiarism_checker.sentence_transformer is not None,
            "advanced_nlp": hasattr(plagiarism_checker, 'transformer_model') and plagiarism_checker.transformer_model is not None
        },
        "assignments_loaded": len(plagiarism_checker.existing_assignments)
    }
