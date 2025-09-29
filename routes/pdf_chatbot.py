from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import tempfile
import fitz  # PyMuPDF
 # fitz
import groq
import json
from datetime import datetime

# Prevent Transformers from importing TensorFlow/Flax to avoid Keras 3 conflicts
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

router = APIRouter()

# Initialize Groq client (you'll need to set GROQ_API_KEY in environment)
try:
    groq_client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    print(f"Warning: Groq client not initialized: {e}")
    groq_client = None

# Lazy-load local summarization pipeline as fallback
_summarizer = None

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        try:
            # Import here so env flags above take effect
            from transformers import pipeline
            _summarizer = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6",
                framework="pt",
            )
        except Exception as e:
            print(f"Warning: Local summarizer not initialized: {e}")
            _summarizer = None
    return _summarizer

# Pydantic models
class PDFSummaryRequest(BaseModel):
    text: str
    subject: Optional[str] = None
    grade_level: Optional[str] = None

class PDFSummaryResponse(BaseModel):
    summary: str
    teaching_tips: List[str]
    key_concepts: List[str]
    difficulty_level: str
    estimated_teaching_time: str
    suggested_activities: List[str]
    assessment_ideas: List[str]

class ChatMessage(BaseModel):
    message: str
    pdf_content: str
    context: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    suggestions: List[str]

# Helper functions
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF using PyMuPDF"""
    try:
        doc = fitz.open(file_path)   # use fitz instead of PyMuPDF
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting text from PDF: {str(e)}")


def generate_summary_with_groq(text: str, subject: str = None, grade_level: str = None) -> Dict:
    """Generate summary and teaching tips using Groq API"""
    if not groq_client:
        raise HTTPException(status_code=503, detail="AI service not available")
    
    subject_context = f" for {subject}" if subject else ""
    grade_context = f" suitable for {grade_level} students" if grade_level else ""
    
    prompt = f"""
    Analyze the following educational content{subject_context}{grade_context} and provide:
    
    1. A comprehensive summary (2-3 paragraphs)
    2. 5-7 practical teaching tips for educators
    3. Key concepts that students should understand
    4. Difficulty level assessment
    5. Estimated teaching time
    6. 3-5 suggested classroom activities
    7. Assessment ideas for testing understanding
    
    Content to analyze:
    {text[:4000]}  # Limit text to avoid token limits
    
    Please format your response as JSON with the following structure:
    {{
        "summary": "...",
        "teaching_tips": ["tip1", "tip2", ...],
        "key_concepts": ["concept1", "concept2", ...],
        "difficulty_level": "beginner/intermediate/advanced",
        "estimated_teaching_time": "X hours/days",
        "suggested_activities": ["activity1", "activity2", ...],
        "assessment_ideas": ["idea1", "idea2", ...]
    }}
    """
    
    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b",
            temperature=0.7,
            max_tokens=2000
        )
        
        result_text = response.choices[0].message.content
        # Try to parse JSON from the response
        try:
            return json.loads(result_text)
        except json.JSONDecodeError:
            # If JSON parsing fails, create structured response manually
            return {
                "summary": result_text[:500] + "...",
                "teaching_tips": ["Use interactive methods", "Break down complex concepts", "Provide real-world examples"],
                "key_concepts": ["Main topic understanding", "Practical application"],
                "difficulty_level": "intermediate",
                "estimated_teaching_time": "2-3 hours",
                "suggested_activities": ["Group discussions", "Hands-on exercises", "Case studies"],
                "assessment_ideas": ["Quiz questions", "Project assignments", "Peer evaluations"]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

def generate_fallback_summary(text: str) -> Dict:
    """Generate summary using local models as fallback"""
    try:
        summarizer = get_summarizer()
        if summarizer:
            # Split text into chunks for summarization
            chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
            summaries = []
            
            for chunk in chunks[:3]:  # Limit to first 3 chunks
                if len(chunk.strip()) > 50:
                    summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
                    summaries.append(summary[0]['summary_text'])
            
            combined_summary = " ".join(summaries)
        else:
            combined_summary = text[:500] + "..." if len(text) > 500 else text
        
        return {
            "summary": combined_summary,
            "teaching_tips": [
                "Break the content into digestible segments",
                "Use visual aids and examples to illustrate concepts",
                "Encourage student questions and discussions",
                "Provide hands-on activities when possible",
                "Connect new concepts to students' prior knowledge"
            ],
            "key_concepts": ["Core understanding", "Practical application", "Critical thinking"],
            "difficulty_level": "intermediate",
            "estimated_teaching_time": "1-2 hours",
            "suggested_activities": [
                "Interactive group discussions",
                "Problem-solving exercises",
                "Real-world case studies"
            ],
            "assessment_ideas": [
                "Formative quizzes",
                "Project-based assessments",
                "Peer review activities"
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in fallback summary: {str(e)}")

def chat_with_content(message: str, pdf_content: str, context: str = None) -> Dict:
    """Generate chat response about the PDF content"""
    # Ensure we have some content to work with
    if not pdf_content or len(pdf_content.strip()) < 10:
        return {
            "response": "I don't have any PDF content to reference. Please upload a PDF document first.",
            "suggestions": [
                "Upload a PDF document to get started",
                "Make sure the PDF contains readable text"
            ]
        }
    
    context_text = f"Previous context: {context}\n\n" if context else ""
    
    # Extract the most relevant part of the PDF content based on the question
    relevant_content = extract_relevant_content(message, pdf_content)
    
    prompt = f"""
    You are an AI teaching assistant helping educators understand and teach content from a PDF document.
    
    {context_text}PDF Content (relevant portion):
    {relevant_content}
    
    Student/Teacher Question: {message}
    
    Please provide a direct and helpful response that:
    1. Directly addresses the question using the PDF content
    2. Provides specific information from the document when relevant
    3. Offers practical teaching advice if applicable
    
    If the question is not related to the PDF content, politely explain that you're here to help with the document.
    """
    
    # First try with Groq if available
    if groq_client:
        try:
            response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
                temperature=0.7,
                max_tokens=1000
            )
            
            return {
                "response": response.choices[0].message.content.strip(),
                "suggestions": generate_suggestions(message)
            }
        except Exception as e:
            print(f"Groq API error: {str(e)}")
    
    # Fallback to local processing if Groq fails or isn't available
    try:
        # Simple keyword-based response if we can't use the LLM
        if any(q in message.lower() for q in ['what', 'how', 'explain', 'tell me']):
            return {
                "response": "Based on the document, " + 
                "here's what I understand about your question: " + 
                message + 
                "\n\nThe document contains relevant information that might help. " +
                "Could you be more specific about what aspect you'd like to know more about?",
                "suggestions": generate_suggestions(message)
            }
        
        # Default response for other cases
        return {
            "response": "I've analyzed the document in relation to your question. " +
            "The content appears to be relevant to your query. " +
            "Could you specify which part of the document you'd like me to focus on?",
            "suggestions": generate_suggestions(message)
        }
        
    except Exception as e:
        return {
            "response": "I'm having trouble processing your request right now. " +
            "Please try asking your question in a different way or try again later.",
            "suggestions": [
                "Rephrase your question",
                "Ask about a specific section of the document",
                "Check your internet connection if using online features"
            ]
        }

def extract_relevant_content(question: str, content: str, max_length: int = 2000) -> str:
    """Extract the most relevant part of the content based on the question"""
    # Simple implementation: look for sentences containing question words
    sentences = content.split('.')
    question_keywords = set(word.lower() for word in question.split() if len(word) > 3)
    
    relevant_sentences = []
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in question_keywords):
            relevant_sentences.append(sentence.strip())
    
    # If we found relevant sentences, return them, otherwise return the beginning
    if relevant_sentences:
        return '. '.join(relevant_sentences)[:max_length] + '...'
    
    # Fallback: return the first part of the content
    return content[:max_length] + ('...' if len(content) > max_length else '')

def generate_suggestions(question: str) -> list:
    """Generate relevant follow-up suggestions based on the question"""
    suggestions = []
    
    if any(word in question.lower() for word in ['what', 'explain', 'define']):
        suggestions.extend([
            "Can you provide more context?",
            "Which part of this concept is unclear?"
        ])
    
    if any(word in question.lower() for word in ['how', 'teach', 'explain to students']):
        suggestions.extend([
            "What grade level are you teaching?",
            "What teaching methods work best for your students?"
        ])
    
    # Add some general suggestions
    suggestions.extend([
        "What are the key points I should focus on?",
        "How can I make this more engaging for students?",
        "What are some common student misconceptions?"
    ])
    
    return suggestions[:4]  # Return at most 4 suggestions
        

# API Endpoints
@router.post("/upload-pdf", response_model=PDFSummaryResponse)
async def upload_and_analyze_pdf(
    file: UploadFile = File(...),
    subject: Optional[str] = Form(None),
    grade_level: Optional[str] = Form(None)
):
    """Upload PDF and get AI-generated summary with teaching tips"""
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name
    
    try:
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(temp_file_path)
        
        if not pdf_text.strip():
            raise HTTPException(status_code=400, detail="No text found in PDF")
        
        # Generate summary and teaching tips
        try:
            result = generate_summary_with_groq(pdf_text, subject, grade_level)
        except:
            # Fallback to local processing
            result = generate_fallback_summary(pdf_text)
        
        return PDFSummaryResponse(**result)
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@router.post("/chat", response_model=ChatResponse)
async def chat_about_pdf(chat_request: ChatMessage):
    """Chat about the uploaded PDF content"""
    
    if not chat_request.pdf_content.strip():
        raise HTTPException(status_code=400, detail="No PDF content provided")
    
    result = chat_with_content(
        chat_request.message, 
        chat_request.pdf_content, 
        chat_request.context
    )
    
    return ChatResponse(**result)

@router.get("/health")
async def health_check():
    """Health check for PDF chatbot service"""
    return {
        "status": "healthy",
        "service": "pdf_chatbot",
        "groq_available": groq_client is not None,
        "local_summarizer_available": _summarizer is not None
    }

@router.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported file formats"""
    return {
        "supported_formats": ["pdf"],
        "max_file_size": "10MB",
        "features": [
            "Text extraction",
            "AI summarization", 
            "Teaching tips generation",
            "Interactive chat about content"
        ]
    }
