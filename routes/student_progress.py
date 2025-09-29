from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import json

router = APIRouter()

# Pydantic models
class StudentMetrics(BaseModel):
    student_id: str
    total_attempts: int
    total_correct: int
    accuracy_rate: float
    total_time_spent: int
    avg_time_per_attempt: float
    topics_studied: List[str]
    performance_trend: str
    overall_score: float
    grade: str

class TopicPerformance(BaseModel):
    topic: str
    attempts: int
    correct: int
    accuracy: float
    time_spent: int
    latest_score: Optional[float] = None
    improvement: float

class StudentProgressData(BaseModel):
    student_info: StudentMetrics
    topic_breakdown: List[TopicPerformance]
    score_history: List[Dict[str, Any]]
    activity_timeline: List[Dict[str, Any]]
    recommendations: List[str]

class ClassOverview(BaseModel):
    total_students: int
    avg_class_performance: float
    top_performers: List[Dict[str, Any]]
    struggling_students: List[Dict[str, Any]]
    topic_difficulty_ranking: List[Dict[str, Any]]

# Helper functions
def load_student_data():
    """Load all student data from CSV files"""
    try:
        base_path = "data"
        
        # Load datasets
        activity_df = pd.read_csv(os.path.join(base_path, "student_activity.csv"))
        scores_df = pd.read_csv(os.path.join(base_path, "student_scores.csv"))
        content_df = pd.read_csv(os.path.join(base_path, "student_content.csv"))
        
        return activity_df, scores_df, content_df
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading student data: {str(e)}")

def calculate_student_metrics(student_id: str, activity_df: pd.DataFrame, scores_df: pd.DataFrame) -> StudentMetrics:
    """Calculate comprehensive metrics for a student"""
    
    # Filter data for the student
    student_activity = activity_df[activity_df['student_id'] == student_id]
    student_scores = scores_df[scores_df['student_id'] == student_id]
    
    if student_activity.empty:
        raise HTTPException(status_code=404, detail=f"No data found for student {student_id}")
    
    # Calculate basic metrics
    total_attempts = student_activity['attempts'].sum()
    total_correct = student_activity['correct'].sum()
    accuracy_rate = (total_correct / total_attempts * 100) if total_attempts > 0 else 0
    total_time_spent = student_activity['time_spent'].sum()
    avg_time_per_attempt = total_time_spent / total_attempts if total_attempts > 0 else 0
    topics_studied = student_activity['topic'].unique().tolist()
    
    # Calculate overall score from test scores
    if not student_scores.empty:
        student_scores['percentage'] = (student_scores['score'] / student_scores['max_score']) * 100
        overall_score = student_scores['percentage'].mean()
    else:
        overall_score = accuracy_rate
    
    # Determine grade
    if overall_score >= 90:
        grade = "A"
    elif overall_score >= 80:
        grade = "B"
    elif overall_score >= 70:
        grade = "C"
    elif overall_score >= 60:
        grade = "D"
    else:
        grade = "F"
    
    # Calculate performance trend (simplified)
    if len(student_scores) >= 2:
        recent_scores = student_scores.tail(2)['percentage'].tolist()
        if recent_scores[-1] > recent_scores[0]:
            performance_trend = "Improving"
        elif recent_scores[-1] < recent_scores[0]:
            performance_trend = "Declining"
        else:
            performance_trend = "Stable"
    else:
        performance_trend = "Insufficient Data"
    
    return StudentMetrics(
        student_id=student_id,
        total_attempts=int(total_attempts),
        total_correct=int(total_correct),
        accuracy_rate=round(accuracy_rate, 1),
        total_time_spent=int(total_time_spent),
        avg_time_per_attempt=round(avg_time_per_attempt, 1),
        topics_studied=topics_studied,
        performance_trend=performance_trend,
        overall_score=round(overall_score, 1),
        grade=grade
    )

def get_topic_performance(student_id: str, activity_df: pd.DataFrame, scores_df: pd.DataFrame) -> List[TopicPerformance]:
    """Get detailed performance breakdown by topic"""
    
    student_activity = activity_df[activity_df['student_id'] == student_id]
    student_scores = scores_df[scores_df['student_id'] == student_id]
    
    topic_performance = []
    
    for topic in student_activity['topic'].unique():
        topic_data = student_activity[student_activity['topic'] == topic]
        
        attempts = topic_data['attempts'].sum()
        correct = topic_data['correct'].sum()
        accuracy = (correct / attempts * 100) if attempts > 0 else 0
        time_spent = topic_data['time_spent'].sum()
        
        # Get latest test score for this topic
        topic_scores = student_scores[student_scores['topic'] == topic]
        latest_score = None
        improvement = 0
        
        if not topic_scores.empty:
            topic_scores = topic_scores.sort_values('date')
            latest_score = (topic_scores.iloc[-1]['score'] / topic_scores.iloc[-1]['max_score']) * 100
            
            # Calculate improvement if multiple scores exist
            if len(topic_scores) >= 2:
                first_score = (topic_scores.iloc[0]['score'] / topic_scores.iloc[0]['max_score']) * 100
                improvement = latest_score - first_score
        
        topic_performance.append(TopicPerformance(
            topic=topic,
            attempts=int(attempts),
            correct=int(correct),
            accuracy=round(accuracy, 1),
            time_spent=int(time_spent),
            latest_score=round(latest_score, 1) if latest_score else None,
            improvement=round(improvement, 1)
        ))
    
    return sorted(topic_performance, key=lambda x: x.accuracy, reverse=True)

def get_score_history(student_id: str, scores_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Get chronological score history for visualization"""
    
    student_scores = scores_df[scores_df['student_id'] == student_id].copy()
    
    if student_scores.empty:
        return []
    
    student_scores['percentage'] = (student_scores['score'] / student_scores['max_score']) * 100
    student_scores = student_scores.sort_values('date')
    
    history = []
    for _, row in student_scores.iterrows():
        history.append({
            "date": row['date'],
            "topic": row['topic'],
            "test_id": row['test_id'],
            "score": int(row['score']),
            "max_score": int(row['max_score']),
            "percentage": round(row['percentage'], 1)
        })
    
    return history

def get_activity_timeline(student_id: str, activity_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Get activity timeline for the student"""
    
    student_activity = activity_df[activity_df['student_id'] == student_id].copy()
    
    timeline = []
    for _, row in student_activity.iterrows():
        timeline.append({
            "topic": row['topic'],
            "attempts": int(row['attempts']),
            "correct": int(row['correct']),
            "time_spent": int(row['time_spent']),
            "accuracy": round((row['correct'] / row['attempts']) * 100, 1) if row['attempts'] > 0 else 0
        })
    
    return sorted(timeline, key=lambda x: x['accuracy'], reverse=True)

def generate_recommendations(student_metrics: StudentMetrics, topic_performance: List[TopicPerformance]) -> List[str]:
    """Generate personalized recommendations for the student"""
    
    recommendations = []
    
    # Performance-based recommendations
    if student_metrics.overall_score < 60:
        recommendations.append("📚 Focus on fundamental concepts - consider reviewing basic materials")
        recommendations.append("👥 Seek help from teachers or tutors for additional support")
    elif student_metrics.overall_score < 80:
        recommendations.append("📈 Good progress! Focus on improving weak areas to reach the next level")
    else:
        recommendations.append("🌟 Excellent performance! Consider taking on more challenging problems")
    
    # Topic-specific recommendations
    weak_topics = [tp for tp in topic_performance if tp.accuracy < 70]
    if weak_topics:
        weakest_topic = min(weak_topics, key=lambda x: x.accuracy)
        recommendations.append(f"🎯 Priority focus needed on {weakest_topic.topic} (current accuracy: {weakest_topic.accuracy}%)")
    
    # Time management recommendations
    if student_metrics.avg_time_per_attempt > 10:
        recommendations.append("⏰ Work on time management - try to solve problems more efficiently")
    elif student_metrics.avg_time_per_attempt < 3:
        recommendations.append("🤔 Take more time to think through problems carefully")
    
    # Trend-based recommendations
    if student_metrics.performance_trend == "Declining":
        recommendations.append("📉 Performance is declining - review recent topics and seek additional help")
    elif student_metrics.performance_trend == "Improving":
        recommendations.append("📈 Great improvement trend! Keep up the good work")
    
    return recommendations

def get_class_overview(activity_df: pd.DataFrame, scores_df: pd.DataFrame) -> ClassOverview:
    """Get overview of entire class performance"""
    
    all_students = activity_df['student_id'].unique()
    student_performances = []
    
    for student_id in all_students:
        try:
            metrics = calculate_student_metrics(student_id, activity_df, scores_df)
            student_performances.append({
                "student_id": student_id,
                "overall_score": metrics.overall_score,
                "grade": metrics.grade,
                "total_attempts": metrics.total_attempts
            })
        except:
            continue
    
    if not student_performances:
        raise HTTPException(status_code=404, detail="No student data available")
    
    # Calculate class statistics
    avg_performance = np.mean([sp["overall_score"] for sp in student_performances])
    
    # Top performers (top 20% or minimum 3)
    sorted_students = sorted(student_performances, key=lambda x: x["overall_score"], reverse=True)
    top_count = max(3, len(sorted_students) // 5)
    top_performers = sorted_students[:top_count]
    
    # Struggling students (bottom 20% or minimum 3, but only if score < 70)
    struggling_students = [s for s in sorted_students if s["overall_score"] < 70]
    struggling_count = max(3, len(struggling_students) // 5)
    struggling_students = struggling_students[-struggling_count:] if struggling_students else []
    
    # Topic difficulty ranking
    topic_stats = activity_df.groupby('topic').agg({
        'attempts': 'sum',
        'correct': 'sum',
        'time_spent': 'mean'
    }).reset_index()
    
    topic_stats['accuracy'] = (topic_stats['correct'] / topic_stats['attempts']) * 100
    topic_difficulty = []
    
    for _, row in topic_stats.iterrows():
        topic_difficulty.append({
            "topic": row['topic'],
            "class_accuracy": round(row['accuracy'], 1),
            "avg_time": round(row['time_spent'], 1),
            "total_attempts": int(row['attempts'])
        })
    
    topic_difficulty = sorted(topic_difficulty, key=lambda x: x['class_accuracy'])
    
    return ClassOverview(
        total_students=len(all_students),
        avg_class_performance=round(avg_performance, 1),
        top_performers=top_performers,
        struggling_students=struggling_students,
        topic_difficulty_ranking=topic_difficulty
    )

# API Endpoints
@router.get("/student/{student_id}", response_model=StudentProgressData)
async def get_student_progress(student_id: str):
    """Get comprehensive progress data for a specific student"""
    
    try:
        activity_df, scores_df, content_df = load_student_data()
        
        # Calculate student metrics
        student_metrics = calculate_student_metrics(student_id, activity_df, scores_df)
        
        # Get topic performance breakdown
        topic_performance = get_topic_performance(student_id, activity_df, scores_df)
        
        # Get score history
        score_history = get_score_history(student_id, scores_df)
        
        # Get activity timeline
        activity_timeline = get_activity_timeline(student_id, activity_df)
        
        # Generate recommendations
        recommendations = generate_recommendations(student_metrics, topic_performance)
        
        return StudentProgressData(
            student_info=student_metrics,
            topic_breakdown=topic_performance,
            score_history=score_history,
            activity_timeline=activity_timeline,
            recommendations=recommendations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting student progress: {str(e)}")

@router.get("/class-overview", response_model=ClassOverview)
async def get_class_overview_data():
    """Get overview of entire class performance"""
    
    try:
        activity_df, scores_df, content_df = load_student_data()
        return get_class_overview(activity_df, scores_df)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting class overview: {str(e)}")

@router.get("/students")
async def get_all_students():
    """Get list of all students with basic info"""
    
    try:
        activity_df, scores_df, content_df = load_student_data()
        
        all_students = activity_df['student_id'].unique()
        student_list = []
        
        for student_id in all_students:
            try:
                metrics = calculate_student_metrics(student_id, activity_df, scores_df)
                student_list.append({
                    "student_id": student_id,
                    "overall_score": metrics.overall_score,
                    "grade": metrics.grade,
                    "topics_studied": len(metrics.topics_studied),
                    "total_attempts": metrics.total_attempts,
                    "performance_trend": metrics.performance_trend
                })
            except:
                continue
        
        return sorted(student_list, key=lambda x: x["overall_score"], reverse=True)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting student list: {str(e)}")

@router.get("/topic-analysis")
async def get_topic_analysis():
    """Get detailed analysis of all topics"""
    
    try:
        activity_df, scores_df, content_df = load_student_data()
        
        # Merge activity with content data for topic details
        topic_analysis = []
        
        for topic in activity_df['topic'].unique():
            topic_activity = activity_df[activity_df['topic'] == topic]
            topic_scores = scores_df[scores_df['topic'] == topic]
            topic_content = content_df[content_df['topic'] == topic]
            
            # Calculate topic statistics
            total_attempts = topic_activity['attempts'].sum()
            total_correct = topic_activity['correct'].sum()
            accuracy = (total_correct / total_attempts * 100) if total_attempts > 0 else 0
            avg_time = topic_activity['time_spent'].mean()
            
            # Get difficulty from content data
            difficulty = topic_content['difficulty'].iloc[0] if not topic_content.empty else "Unknown"
            
            # Calculate average test score
            avg_test_score = 0
            if not topic_scores.empty:
                topic_scores['percentage'] = (topic_scores['score'] / topic_scores['max_score']) * 100
                avg_test_score = topic_scores['percentage'].mean()
            
            topic_analysis.append({
                "topic": topic,
                "difficulty_level": difficulty,
                "students_attempted": len(topic_activity['student_id'].unique()),
                "total_attempts": int(total_attempts),
                "accuracy_rate": round(accuracy, 1),
                "avg_time_spent": round(avg_time, 1),
                "avg_test_score": round(avg_test_score, 1),
                "performance_rating": "High" if accuracy > 80 else "Medium" if accuracy > 60 else "Low"
            })
        
        return sorted(topic_analysis, key=lambda x: x['accuracy_rate'], reverse=True)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting topic analysis: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check for student progress service"""
    return {
        "status": "healthy",
        "service": "student_progress",
        "data_files_available": ["student_activity.csv", "student_scores.csv", "student_content.csv"]
    }
