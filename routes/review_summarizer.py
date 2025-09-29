from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import os
import tempfile
import csv
import io
from textblob import TextBlob
import groq
import json
from collections import Counter

router = APIRouter()

# Initialize Groq client
try:
    groq_client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    print(f"Warning: Groq client not initialized: {e}")
    groq_client = None

# Pydantic models
class ReviewAnalysis(BaseModel):
    total_reviews: int
    positive_count: int
    negative_count: int
    neutral_count: int
    positive_percentage: float
    negative_percentage: float
    neutral_percentage: float
    sentiment_distribution: Dict[str, int]

class ImprovementSuggestion(BaseModel):
    category: str
    issues: List[str]
    suggestions: List[str]
    priority: str

class ReviewSummaryResponse(BaseModel):
    analysis: ReviewAnalysis
    improvement_suggestions: List[ImprovementSuggestion]
    key_themes: Dict[str, List[str]]
    overall_rating: float
    summary: str

class ReviewData(BaseModel):
    reviews: List[str]
    ratings: Optional[List[int]] = None

# Helper functions
def analyze_sentiment_textblob(text: str) -> Dict[str, float]:
    """Analyze sentiment using TextBlob"""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    if polarity > 0.1:
        sentiment = "positive"
    elif polarity < -0.1:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    return {
        "sentiment": sentiment,
        "polarity": polarity,
        "subjectivity": blob.sentiment.subjectivity
    }

def categorize_reviews(reviews: List[str]) -> Dict:
    """Categorize reviews into positive, negative, and neutral"""
    results = {
        "positive": [],
        "negative": [],
        "neutral": [],
        "sentiments": []
    }
    
    for review in reviews:
        if not review.strip():
            continue
            
        sentiment_data = analyze_sentiment_textblob(review)
        sentiment = sentiment_data["sentiment"]
        
        results[sentiment].append({
            "text": review,
            "polarity": sentiment_data["polarity"],
            "subjectivity": sentiment_data["subjectivity"]
        })
        results["sentiments"].append(sentiment)
    
    return results

def extract_themes_and_issues(negative_reviews: List[Dict]) -> Dict[str, List[str]]:
    """Extract common themes from negative reviews"""
    themes = {
        "teaching_methods": [],
        "communication": [],
        "course_content": [],
        "assessment": [],
        "classroom_management": [],
        "general": []
    }
    
    # Keywords for categorization
    keywords = {
        "teaching_methods": ["boring", "unclear", "confusing", "difficult", "hard to understand", "poor explanation", "bad teaching", "doesn't explain well"],
        "communication": ["rude", "unfriendly", "doesn't respond", "poor communication", "dismissive", "not helpful", "ignores questions"],
        "course_content": ["outdated", "irrelevant", "too much", "too little", "disorganized", "boring content", "not useful"],
        "assessment": ["unfair", "too hard", "unclear instructions", "grading", "feedback", "harsh grading", "no feedback"],
        "classroom_management": ["disruptive", "noise", "time management", "late", "unprepared", "disorganized class", "poor management"]
    }
    
    for review_data in negative_reviews:
        review_text = review_data["text"].lower()
        categorized = False
        
        for category, category_keywords in keywords.items():
            for keyword in category_keywords:
                if keyword in review_text:
                    themes[category].append(review_text)
                    categorized = True
                    break
            if categorized:
                break
        
        if not categorized:
            themes["general"].append(review_text)
    
    # Ensure we always return some suggestions even if no specific themes are found
    if all(not themes[key] for key in themes.keys()):
        # If no themes found, add general negative reviews to general category
        for review_data in negative_reviews:
            themes["general"].append(review_data["text"].lower())
    
    return themes

def generate_improvement_suggestions_with_ai(negative_themes: Dict[str, List[str]]) -> List[ImprovementSuggestion]:
    """Generate improvement suggestions using AI"""
    suggestions = []
    
    # Fallback suggestions
    fallback_suggestions = {
        "teaching_methods": ImprovementSuggestion(
            category="Teaching Methods",
            issues=["Unclear explanations", "Boring delivery"],
            suggestions=[
                "Use more interactive teaching methods like group discussions",
                "Incorporate visual aids and real-world examples",
                "Break complex topics into smaller, digestible parts",
                "Ask for student feedback during lessons"
            ],
            priority="high"
        ),
        "communication": ImprovementSuggestion(
            category="Communication",
            issues=["Poor student interaction", "Lack of responsiveness"],
            suggestions=[
                "Establish regular office hours for student questions",
                "Respond to emails within 24-48 hours",
                "Create a more welcoming classroom environment",
                "Practice active listening techniques"
            ],
            priority="high"
        ),
        "course_content": ImprovementSuggestion(
            category="Course Content",
            issues=["Outdated material", "Poor organization"],
            suggestions=[
                "Update course materials regularly",
                "Prepare materials in advance",
                "Establish clear classroom rules"
            ],
            priority="low"
        ),
        "assessment": ImprovementSuggestion(
            category="Assessment",
            issues=["Unclear grading", "Unfair tests"],
            suggestions=[
                "Provide clear rubrics for assignments",
                "Give timely and constructive feedback",
                "Align assessments with learning objectives",
                "Offer practice tests or examples"
            ],
            priority="medium"
        ),
        "classroom_management": ImprovementSuggestion(
            category="Classroom Management",
            issues=["Poor time management", "Disorganization"],
            suggestions=[
                "Create and stick to a structured lesson plan",
                "Start and end classes on time",
                "Prepare materials in advance",
                "Establish clear classroom rules"
            ],
            priority="low"
        ),
        "general": ImprovementSuggestion(
            category="General Improvements",
            issues=["Various concerns identified"],
            suggestions=[
                "Seek regular feedback from students",
                "Attend professional development workshops",
                "Collaborate with other teachers for best practices",
                "Reflect on teaching methods and adjust as needed"
            ],
            priority="medium"
        )
    }
    
    # Always provide suggestions if we have negative themes
    has_negative_feedback = any(themes for themes in negative_themes.values())
    
    print(f"DEBUG: has_negative_feedback = {has_negative_feedback}")
    print(f"DEBUG: negative_themes = {negative_themes}")
    print(f"DEBUG: groq_client available = {groq_client is not None}")
    
    # If no negative feedback, return empty suggestions
    if not has_negative_feedback:
        return suggestions
    
    # Always use fallback suggestions first (works with or without AI)
    for category, issues in negative_themes.items():
        if issues and category in fallback_suggestions:
            print(f"DEBUG: Adding suggestion for category {category} with {len(issues)} issues")
            suggestions.append(fallback_suggestions[category])
    
    # If no specific categories found but we have general negative feedback
    if not suggestions and has_negative_feedback:
        print("DEBUG: No specific categories found, adding general suggestions")
        suggestions.append(fallback_suggestions["general"])
    
    print(f"DEBUG: Total suggestions generated: {len(suggestions)}")
    
    # Return fallback suggestions (skip AI for now to ensure basic functionality works)
    return suggestions
    
    # AI-powered suggestions
    for category, issues in negative_themes.items():
        if not issues:
            continue
        
        issues_text = "\n".join(issues[:5])  # Limit to first 5 issues
        
        prompt = f"""
        Analyze the following negative feedback about a teacher in the category "{category}":
        
        {issues_text}
        
        Provide specific, actionable improvement suggestions. Format as JSON:
        {{
            "category": "{category}",
            "issues": ["issue1", "issue2"],
            "suggestions": ["suggestion1", "suggestion2", "suggestion3"],
            "priority": "high/medium/low"
        }}
        """
        
        try:
            response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
                temperature=0.7,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content
            try:
                suggestion_data = json.loads(result_text)
                suggestions.append(ImprovementSuggestion(**suggestion_data))
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                suggestions.append(ImprovementSuggestion(
                    category=category.replace("_", " ").title(),
                    issues=["Various issues identified"],
                    suggestions=["Review and improve based on feedback"],
                    priority="medium"
                ))
        except Exception as e:
            print(f"Error generating AI suggestions for {category}: {e}")
    
    return suggestions

def parse_csv_file(file_content: bytes) -> List[str]:
    """Parse CSV file and extract review text"""
    reviews = []
    try:
        content = file_content.decode('utf-8')
        csv_reader = csv.reader(io.StringIO(content))
        headers = next(csv_reader, None)
        
        # Try to find review column (common names)
        review_column_names = ['review', 'feedback', 'comment', 'comments', 'text', 'response']
        review_col_index = None
        
        if headers:
            headers_lower = [h.lower().strip() for h in headers]
            for col_name in review_column_names:
                if col_name in headers_lower:
                    review_col_index = headers_lower.index(col_name)
                    break
        
        if review_col_index is None:
            # If no specific column found, use the first text column
            review_col_index = 0
        
        for row in csv_reader:
            if len(row) > review_col_index and row[review_col_index].strip():
                reviews.append(row[review_col_index].strip())
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV file: {str(e)}")
    
    return reviews

# API Endpoints
@router.post("/analyze-reviews", response_model=ReviewSummaryResponse)
async def analyze_teacher_reviews(file: UploadFile = File(...)):
    """Analyze teacher performance reviews from uploaded CSV file"""
    
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        # Read file content
        content = await file.read()
        reviews = parse_csv_file(content)
        
        if not reviews:
            raise HTTPException(status_code=400, detail="No reviews found in the file")
        
        # Categorize reviews
        categorized = categorize_reviews(reviews)
        
        # Calculate statistics
        total_reviews = len(reviews)
        positive_count = len(categorized["positive"])
        negative_count = len(categorized["negative"])
        neutral_count = len(categorized["neutral"])
        
        analysis = ReviewAnalysis(
            total_reviews=total_reviews,
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            positive_percentage=round((positive_count / total_reviews) * 100, 1),
            negative_percentage=round((negative_count / total_reviews) * 100, 1),
            neutral_percentage=round((neutral_count / total_reviews) * 100, 1),
            sentiment_distribution=Counter(categorized["sentiments"])
        )
        
        # Extract themes from negative reviews
        negative_themes = extract_themes_and_issues(categorized["negative"])
        
        # Generate improvement suggestions
        improvement_suggestions = generate_improvement_suggestions_with_ai(negative_themes)
        print(f"Generated {len(improvement_suggestions)} improvement suggestions")
        print(f"Negative themes: {negative_themes}")
        
        # Calculate overall rating (1-5 scale)
        avg_polarity = np.mean([
            review["polarity"] for category in ["positive", "negative", "neutral"]
            for review in categorized[category]
        ])
        overall_rating = round(((avg_polarity + 1) / 2) * 5, 1)  # Convert from [-1,1] to [0,5]
        
        # Generate summary
        summary = f"Analysis of {total_reviews} teacher reviews shows {analysis.positive_percentage}% positive, {analysis.negative_percentage}% negative, and {analysis.neutral_percentage}% neutral feedback. Overall rating: {overall_rating}/5."
        
        return ReviewSummaryResponse(
            analysis=analysis,
            improvement_suggestions=improvement_suggestions,
            key_themes=negative_themes,
            overall_rating=overall_rating,
            summary=summary
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing reviews: {str(e)}")

@router.post("/analyze-text-reviews", response_model=ReviewSummaryResponse)
async def analyze_text_reviews(review_data: ReviewData):
    """Analyze teacher reviews from direct text input"""
    
    if not review_data.reviews:
        raise HTTPException(status_code=400, detail="No reviews provided")
    
    try:
        # Categorize reviews
        categorized = categorize_reviews(review_data.reviews)
        
        # Calculate statistics
        total_reviews = len(review_data.reviews)
        positive_count = len(categorized["positive"])
        negative_count = len(categorized["negative"])
        neutral_count = len(categorized["neutral"])
        
        analysis = ReviewAnalysis(
            total_reviews=total_reviews,
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            positive_percentage=round((positive_count / total_reviews) * 100, 1),
            negative_percentage=round((negative_count / total_reviews) * 100, 1),
            neutral_percentage=round((neutral_count / total_reviews) * 100, 1),
            sentiment_distribution=Counter(categorized["sentiments"])
        )
        
        # Extract themes from negative reviews
        negative_themes = extract_themes_and_issues(categorized["negative"])
        
        # Generate improvement suggestions
        improvement_suggestions = generate_improvement_suggestions_with_ai(negative_themes)
        print(f"Text reviews - Generated {len(improvement_suggestions)} improvement suggestions")
        print(f"Text reviews - Negative themes: {negative_themes}")
        
        # Calculate overall rating
        if review_data.ratings:
            overall_rating = round(np.mean(review_data.ratings), 1)
        else:
            avg_polarity = np.mean([
                review["polarity"] for category in ["positive", "negative", "neutral"]
                for review in categorized[category]
            ])
            overall_rating = round(((avg_polarity + 1) / 2) * 5, 1)
        
        # Generate summary
        summary = (
    f"Analysis of {total_reviews} teacher reviews shows "
    f"{analysis.positive_percentage}% positive, "
    f"{analysis.negative_percentage}% negative, "
    f"{analysis.neutral_percentage}% neutral feedback. "
    f"Overall rating: {overall_rating}/5."
)

        
        return ReviewSummaryResponse(
            analysis=analysis,
            improvement_suggestions=improvement_suggestions,
            key_themes=negative_themes,
            overall_rating=overall_rating,
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing reviews: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check for review summarizer service"""
    return {
        "status": "healthy",
        "service": "review_summarizer",
        "ai_available": groq_client is not None
    }

@router.get("/supported-formats")
async def get_supported_formats():
    """Get supported file formats and expected structure"""
    return {
        "supported_formats": ["csv"],
        "expected_columns": ["review", "feedback", "comment", "comments", "text", "response"],
        "max_file_size": "5MB",
        "features": [
            "Sentiment analysis",
            "Review categorization",
            "Improvement suggestions",
            "Theme extraction"
        ]
    }
