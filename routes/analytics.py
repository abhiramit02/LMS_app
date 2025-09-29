from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Optional
import sys
import os

# Add parent directory to path to import student_analytics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from student_analytics import StudentPerformanceAnalytics

router = APIRouter()

# Initialize analytics system
try:
    analytics_system = StudentPerformanceAnalytics()
    print("Student analytics system initialized successfully")
except Exception as e:
    print(f"Error initializing analytics system: {e}")
    analytics_system = None

# Pydantic models for API requests/responses
class StudentMetrics(BaseModel):
    student_id: str
    total_time_spent: float
    avg_score: float
    completion_rate: float
    consistency_score: float
    improvement_trend: float
    engagement_level: str
    performance_category: str
    risk_level: str
    risk_score: float
    topics_attempted: int
    total_attempts: int

class AtRiskStudent(BaseModel):
    student_id: str
    risk_score: float
    avg_score: float
    completion_rate: float
    engagement_level: str
    main_issues: List[str]

class ClusterInfo(BaseModel):
    size: int
    avg_score: float
    avg_completion_rate: float
    avg_engagement: float
    dominant_risk_level: str
    characteristics: str

class Recommendation(BaseModel):
    priority_actions: List[str]
    study_tips: List[str]
    resource_suggestions: List[str]
    timeline: str
    focus_areas: List[str]

class PerformanceInsights(BaseModel):
    total_students: int
    performance_distribution: Dict[str, int]
    risk_distribution: Dict[str, int]
    engagement_distribution: Dict[str, int]
    overall_metrics: Dict[str, float]
    trends: Dict[str, int]

class AnalyticsDashboard(BaseModel):
    overview: PerformanceInsights
    at_risk_students: Dict[str, List[AtRiskStudent]]
    learning_clusters: Dict
    total_critical_risk: int
    total_high_risk: int
    improvement_rate: float

# API Endpoints
@router.get("/health")
async def health_check():
    """Health check for analytics service"""
    if analytics_system is None:
        raise HTTPException(status_code=503, detail="Analytics system not available")
    return {"status": "healthy", "service": "student_analytics"}

@router.get("/overview")
async def get_analytics_overview():
    """Get comprehensive analytics overview for all students"""
    if analytics_system is None:
        raise HTTPException(status_code=503, detail="Analytics system not available")
    
    try:
        dashboard_data = analytics_system.get_student_dashboard_data()
        
        # Calculate additional metrics
        critical_risk = len(dashboard_data['at_risk_students']['critical'])
        high_risk = len(dashboard_data['at_risk_students']['high'])
        total_students = dashboard_data['overview']['total_students']
        
        improvement_rate = 0
        if total_students > 0:
            improving = dashboard_data['overview']['trends']['improving_students']
            improvement_rate = (improving / total_students) * 100
        
        return {
            'overview': dashboard_data['overview'],
            'at_risk_students': dashboard_data['at_risk_students'],
            'learning_clusters': dashboard_data['learning_clusters'],
            'total_critical_risk': critical_risk,
            'total_high_risk': high_risk,
            'improvement_rate': improvement_rate,
            'cluster_analysis': dashboard_data['learning_clusters']['cluster_info']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting analytics overview: {str(e)}")

@router.get("/student/{student_id}")
async def get_student_analytics(student_id: str):
    """Get detailed analytics for a specific student"""
    if analytics_system is None:
        raise HTTPException(status_code=503, detail="Analytics system not available")
    
    try:
        student_data = analytics_system.get_student_dashboard_data(student_id)
        
        if 'error' in student_data:
            raise HTTPException(status_code=404, detail=student_data['error'])
        
        return student_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting student analytics: {str(e)}")

@router.get("/at-risk")
async def get_at_risk_students(
    risk_level: Optional[str] = Query(None, description="Filter by risk level: critical, high, medium, low")
):
    """Get list of at-risk students"""
    if analytics_system is None:
        raise HTTPException(status_code=503, detail="Analytics system not available")
    
    try:
        at_risk_data = analytics_system.at_risk_students
        
        if risk_level and risk_level in at_risk_data:
            return {risk_level: at_risk_data[risk_level]}
        
        return at_risk_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting at-risk students: {str(e)}")

@router.get("/clusters")
async def get_learning_clusters():
    """Get learning pattern clusters"""
    if analytics_system is None:
        raise HTTPException(status_code=503, detail="Analytics system not available")
    
    try:
        return analytics_system.learning_clusters
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting learning clusters: {str(e)}")

@router.get("/recommendations/{student_id}")
async def get_student_recommendations(student_id: str):
    """Get personalized recommendations for a student"""
    if analytics_system is None:
        raise HTTPException(status_code=503, detail="Analytics system not available")
    
    try:
        student_id = student_id.lower()
        
        if student_id not in analytics_system.improvement_recommendations:
            raise HTTPException(status_code=404, detail=f"No recommendations found for student {student_id}")
        
        return analytics_system.improvement_recommendations[student_id]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")

@router.get("/performance-trends")
async def get_performance_trends():
    """Get performance trends analysis"""
    if analytics_system is None:
        raise HTTPException(status_code=503, detail="Analytics system not available")
    
    try:
        insights = analytics_system.performance_insights
        
        # Calculate trend percentages
        total = insights['total_students']
        trends = insights['trends']
        
        trend_analysis = {
            'improving_percentage': (trends['improving_students'] / total * 100) if total > 0 else 0,
            'declining_percentage': (trends['declining_students'] / total * 100) if total > 0 else 0,
            'stable_percentage': (trends['stable_students'] / total * 100) if total > 0 else 0,
            'total_students': total,
            'raw_counts': trends
        }
        
        return trend_analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting performance trends: {str(e)}")

@router.get("/intervention-priorities")
async def get_intervention_priorities():
    """Get prioritized list of students needing intervention"""
    if analytics_system is None:
        raise HTTPException(status_code=503, detail="Analytics system not available")
    
    try:
        at_risk = analytics_system.at_risk_students
        
        # Combine critical and high risk students
        priority_students = []
        
        # Add critical risk students first
        for student in at_risk['critical']:
            student['priority'] = 'CRITICAL'
            priority_students.append(student)
        
        # Add high risk students
        for student in at_risk['high']:
            student['priority'] = 'HIGH'
            priority_students.append(student)
        
        # Sort by risk score (ascending - most at risk first)
        priority_students.sort(key=lambda x: x['risk_score'])
        
        return {
            'total_priority_students': len(priority_students),
            'critical_count': len(at_risk['critical']),
            'high_risk_count': len(at_risk['high']),
            'priority_list': priority_students
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting intervention priorities: {str(e)}")

@router.post("/refresh-analytics")
async def refresh_analytics():
    """Refresh analytics data by reprocessing all student data"""
    global analytics_system
    
    try:
        analytics_system = StudentPerformanceAnalytics()
        return {"status": "success", "message": "Analytics data refreshed successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error refreshing analytics: {str(e)}")

@router.get("/export-report")
async def export_analytics_report():
    """Export comprehensive analytics report"""
    if analytics_system is None:
        raise HTTPException(status_code=503, detail="Analytics system not available")
    
    try:
        output_path = analytics_system.export_analytics_report()
        
        if output_path:
            return {
                "status": "success", 
                "message": "Analytics report exported successfully",
                "file_path": output_path
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to export analytics report")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting report: {str(e)}")

@router.get("/stats")
async def get_system_stats():
    """Get analytics system statistics"""
    if analytics_system is None:
        raise HTTPException(status_code=503, detail="Analytics system not available")
    
    try:
        stats = {
            "total_students_analyzed": len(analytics_system.student_metrics),
            "data_sources": {
                "activity_records": len(analytics_system.activity_df),
                "score_records": len(analytics_system.scores_df),
                "content_items": len(analytics_system.content_df)
            },
            "risk_summary": {
                "critical": len(analytics_system.at_risk_students['critical']),
                "high": len(analytics_system.at_risk_students['high']),
                "medium": len(analytics_system.at_risk_students['medium']),
                "low": len(analytics_system.at_risk_students['low'])
            },
            "clusters_identified": len(analytics_system.learning_clusters['clusters']),
            "recommendations_generated": len(analytics_system.improvement_recommendations)
        }
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting system stats: {str(e)}")
