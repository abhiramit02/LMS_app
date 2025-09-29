from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Optional
import sys
import os

# Add parent directory to path to import recommendation_system
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Lazy/safe import to avoid crashing on serverless when heavy deps are absent
try:
    from recommendation_system import LMSRecommendationSystem
    _RECSYS_IMPORT_OK = True
except Exception as e:
    print(f"Warning: recommendation_system import failed: {e}")
    LMSRecommendationSystem = None  # type: ignore
    _RECSYS_IMPORT_OK = False

router = APIRouter()

# Initialize recommendation system (only if import succeeded)
try:
    rec_system = LMSRecommendationSystem() if _RECSYS_IMPORT_OK else None
    if rec_system is not None:
        print("Recommendation system initialized successfully")
except Exception as e:
    print(f"Error initializing recommendation system: {e}")
    rec_system = None

# Pydantic models for API requests/responses
class StudentProfile(BaseModel):
    student_id: str
    total_time_spent: float
    avg_completion_rate: float
    total_attempts: int
    avg_score: float
    topics_attempted: int
    courses_enrolled: int
    preferred_time: Optional[int] = None
    avg_session_duration: Optional[float] = None

class Recommendation(BaseModel):
    topic_id: str
    predicted_score: float
    confidence: float
    method: str
    cf_score: Optional[float] = 0
    cb_score: Optional[float] = 0
    tf_score: Optional[float] = 0
    final_score: Optional[float] = 0
    methods: Optional[List[str]] = []

class NextCourse(BaseModel):
    topic_id: str
    title: str
    description: str
    difficulty: str
    duration: float
    readiness_score: float
    prerequisites: List[str]

class LearningMaterial(BaseModel):
    type: str
    content: str
    difficulty: str
    reason: str

class LearningPathItem(BaseModel):
    topic_id: str
    status: str
    score: Optional[float] = None
    readiness_score: Optional[float] = None
    difficulty: Optional[str] = None
    order: int

class PersonalizedDashboard(BaseModel):
    profile: StudentProfile
    recommendations: List[Recommendation]
    next_courses: List[NextCourse]
    progress: Dict[str, Dict]
    strengths: List[str]
    weaknesses: List[str]
    learning_path: List[LearningPathItem]

class ProgressData(BaseModel):
    avg_score: float
    attempts: int
    last_attempt: float

# API Endpoints
@router.get("/health")
async def health_check():
    """Health check for recommendation service"""
    if rec_system is None:
        raise HTTPException(status_code=503, detail="Recommendation system not available")
    return {"status": "healthy", "service": "recommendation_system"}

@router.get("/profile/{student_id}", response_model=StudentProfile)
async def get_student_profile(student_id: str):
    """Get comprehensive student profile"""
    if rec_system is None:
        raise HTTPException(status_code=503, detail="Recommendation system not available")
    
    try:
        profile = rec_system.get_student_profile(student_id)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Student {student_id} not found")
        
        return StudentProfile(**profile)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting profile: {str(e)}")

@router.get("/collaborative-filtering/{student_id}", response_model=List[Recommendation])
async def get_collaborative_filtering_recommendations(
    student_id: str,
    n_recommendations: int = Query(5, ge=1, le=20, description="Number of recommendations")
):
    """Get recommendations using collaborative filtering"""
    if rec_system is None:
        raise HTTPException(status_code=503, detail="Recommendation system not available")
    
    try:
        recommendations = rec_system.collaborative_filtering_recommendations(student_id, n_recommendations)
        return [Recommendation(**rec) for rec in recommendations]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting collaborative filtering recommendations: {str(e)}")

@router.get("/content-based/{student_id}", response_model=List[Recommendation])
async def get_content_based_recommendations(
    student_id: str,
    n_recommendations: int = Query(5, ge=1, le=20, description="Number of recommendations")
):
    """Get recommendations using content-based filtering"""
    if rec_system is None:
        raise HTTPException(status_code=503, detail="Recommendation system not available")
    
    try:
        recommendations = rec_system.content_based_filtering(student_id, n_recommendations)
        return [Recommendation(**rec) for rec in recommendations]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting content-based recommendations: {str(e)}")

@router.get("/transformer-based/{student_id}", response_model=List[Recommendation])
async def get_transformer_based_recommendations(
    student_id: str,
    n_recommendations: int = Query(5, ge=1, le=20, description="Number of recommendations")
):
    """Get recommendations using transformer models"""
    if rec_system is None:
        raise HTTPException(status_code=503, detail="Recommendation system not available")
    
    try:
        recommendations = rec_system.transformer_based_recommendations(student_id, n_recommendations)
        return [Recommendation(**rec) for rec in recommendations]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting transformer-based recommendations: {str(e)}")

@router.get("/hybrid/{student_id}", response_model=List[Recommendation])
async def get_hybrid_recommendations(
    student_id: str,
    n_recommendations: int = Query(10, ge=1, le=50, description="Number of recommendations")
):
    """Get hybrid recommendations combining all methods"""
    if rec_system is None:
        raise HTTPException(status_code=503, detail="Recommendation system not available")
    
    try:
        recommendations = rec_system.hybrid_recommendations(student_id, n_recommendations)
        return [Recommendation(**rec) for rec in recommendations]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting hybrid recommendations: {str(e)}")

@router.get("/next-courses/{student_id}", response_model=List[NextCourse])
async def get_next_course_recommendations(
    student_id: str,
    n_recommendations: int = Query(5, ge=1, le=20, description="Number of recommendations")
):
    """Get next course/module recommendations based on student progress"""
    if rec_system is None:
        raise HTTPException(status_code=503, detail="Recommendation system not available")
    
    try:
        next_courses = rec_system.get_next_course_recommendations(student_id, n_recommendations)
        return [NextCourse(**course) for course in next_courses]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting next course recommendations: {str(e)}")

@router.get("/learning-materials/{student_id}/{topic_id}", response_model=List[LearningMaterial])
async def get_learning_material_recommendations(
    student_id: str,
    topic_id: str,
    n_recommendations: int = Query(5, ge=1, le=20, description="Number of recommendations")
):
    """Get additional learning material recommendations for a specific topic"""
    if rec_system is None:
        raise HTTPException(status_code=503, detail="Recommendation system not available")
    
    try:
        materials = rec_system.get_learning_material_recommendations(student_id, topic_id, n_recommendations)
        return [LearningMaterial(**material) for material in materials]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting learning material recommendations: {str(e)}")

@router.get("/dashboard/{student_id}")
async def get_personalized_dashboard(student_id: str):
    """Get personalized dashboard data for a student"""
    if rec_system is None:
        raise HTTPException(status_code=503, detail="Recommendation system not available")
    
    try:
        print(f"DEBUG: Getting dashboard for student: {student_id}")
        dashboard_data = rec_system.get_personalized_dashboard_data(student_id)
        print(f"DEBUG: Dashboard data keys: {list(dashboard_data.keys()) if dashboard_data else 'None'}")
        
        if not dashboard_data:
            raise HTTPException(status_code=404, detail=f"Dashboard data not found for student {student_id}")
        
        # Return raw data without Pydantic validation for now
        return dashboard_data
    except Exception as e:
        print(f"DEBUG: Error in dashboard endpoint: {str(e)}")
        print(f"DEBUG: Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error getting personalized dashboard: {str(e)}")

@router.get("/learning-path/{student_id}", response_model=List[LearningPathItem])
async def get_learning_path(student_id: str):
    """Get personalized learning path for a student"""
    if rec_system is None:
        raise HTTPException(status_code=503, detail="Recommendation system not available")
    
    try:
        learning_path = rec_system.generate_learning_path(student_id)
        return [LearningPathItem(**item) for item in learning_path]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating learning path: {str(e)}")

@router.get("/progress/{student_id}")
async def get_student_progress(student_id: str):
    """Get detailed progress data for a student"""
    if rec_system is None:
        raise HTTPException(status_code=503, detail="Recommendation system not available")
    
    try:
        student_scores = rec_system.student_scores[rec_system.student_scores['student_id'] == student_id]
        
        if student_scores.empty:
            raise HTTPException(status_code=404, detail=f"No progress data found for student {student_id}")
        
        progress_data = {}
        for topic_id in student_scores['topic_id'].unique():
            topic_scores = student_scores[student_scores['topic_id'] == topic_id]
            progress_data[topic_id] = {
                'avg_score': float(topic_scores['score'].mean()),
                'attempts': int(topic_scores['attempts'].sum()),
                'last_attempt': float(topic_scores['score'].iloc[-1]) if len(topic_scores) > 0 else 0,
                'total_score': float(topic_scores['score'].sum()),
                'max_possible': float(topic_scores['max_score'].sum()) if 'max_score' in topic_scores.columns else 0
            }
        
        return {
            "student_id": student_id,
            "progress": progress_data,
            "overall_stats": {
                "total_topics": len(progress_data),
                "avg_score": float(student_scores['score'].mean()),
                "total_attempts": int(student_scores['attempts'].sum()),
                "completion_rate": len([p for p in progress_data.values() if p['avg_score'] >= 70]) / len(progress_data) if progress_data else 0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting progress data: {str(e)}")

@router.post("/reload-data")
async def reload_data():
    """Reload data and reinitialize the recommendation system"""
    global rec_system
    
    try:
        rec_system = LMSRecommendationSystem()
        return {"status": "success", "message": "Recommendation system reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading data: {str(e)}")

@router.post("/save-models")
async def save_models():
    """Save trained models for later use"""
    if rec_system is None:
        raise HTTPException(status_code=503, detail="Recommendation system not available")
    
    try:
        rec_system.save_models()
        return {"status": "success", "message": "Models saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving models: {str(e)}")

@router.post("/load-models")
async def load_models():
    """Load previously saved models"""
    if rec_system is None:
        raise HTTPException(status_code=503, detail="Recommendation system not available")
    
    try:
        rec_system.load_models()
        return {"status": "success", "message": "Models loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")

@router.get("/stats")
async def get_system_stats():
    """Get system statistics and status"""
    if rec_system is None:
        raise HTTPException(status_code=503, detail="Recommendation system not available")
    
    try:
        stats = {
            "total_students": rec_system.student_scores['student_id'].nunique() if not rec_system.student_scores.empty else 0,
            "total_topics": rec_system.student_content['topic_id'].nunique() if not rec_system.student_content.empty else 0,
            "total_questions": len(rec_system.question_bank) if not rec_system.question_bank.empty else 0,
            "total_activities": len(rec_system.student_activity) if not rec_system.student_activity.empty else 0,
            "transformers_available": rec_system.transformer_model is not None,
            "surprise_available": hasattr(rec_system, 'SURPRISE_AVAILABLE') and rec_system.SURPRISE_AVAILABLE,
            "data_loaded": all([
                not rec_system.student_activity.empty,
                not rec_system.student_scores.empty,
                not rec_system.student_content.empty
            ])
        }
        
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting system stats: {str(e)}")
