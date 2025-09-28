import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging

# Text processing and NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams

# ML and similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

# File parsing
import pdfplumber
from docx import Document
import PyPDF2
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Advanced NLP
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    from sentence_transformers import SentenceTransformer
    ADVANCED_NLP_AVAILABLE = True
except ImportError:
    ADVANCED_NLP_AVAILABLE = False
    print("Warning: Advanced NLP libraries not available. Using basic methods.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlagiarismChecker:
    """
    Advanced plagiarism detection system using multiple techniques:
    - Text similarity analysis
    - Semantic similarity using transformers
    - Fingerprinting and hashing
    - Cross-reference checking
    - Statistical analysis
    """
    
    def __init__(self, data_dir: str = "data", assignments_dir: str = "data/assignments"):
        self.data_dir = Path(data_dir)
        self.assignments_dir = Path(assignments_dir)
        self.assignments_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize NLTK
        self._initialize_nltk()
        
        # Initialize models
        self.tfidf_vectorizer = None
        self.sentence_transformer = None
        self.transformer_model = None
        self.transformer_tokenizer = None
        
        # Load existing assignments for comparison
        self.existing_assignments = {}
        self.assignment_hashes = {}
        self.load_existing_assignments()
        
        # Initialize models
        self._initialize_models()
        
        # Plagiarism thresholds
        self.similarity_threshold = 0.7
        self.semantic_threshold = 0.75
        self.hash_threshold = 0.8
        
    def _initialize_nltk(self):
        """Initialize NLTK resources"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except Exception as e:
            logger.warning(f"Failed to download NLTK resources: {e}")
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def _initialize_models(self):
        """Initialize ML models for plagiarism detection"""
        try:
            # Initialize TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=2
            )
            logger.info("TF-IDF vectorizer initialized successfully")
            
            # Initialize sentence transformer if available
            if ADVANCED_NLP_AVAILABLE:
                try:
                    self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
                    logger.info("Sentence transformer initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize sentence transformer: {e}")
                
                # Initialize transformer model for semantic analysis
                try:
                    self.transformer_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                    self.transformer_model = AutoModel.from_pretrained('bert-base-uncased')
                    logger.info("Transformer model initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize transformer model: {e}")
            else:
                logger.info("Advanced NLP not available, using basic TF-IDF only")
                    
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            # Ensure TF-IDF is available as fallback
            try:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=2000,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=1
                )
                logger.info("Fallback TF-IDF vectorizer initialized")
            except Exception as fallback_e:
                logger.error(f"Failed to initialize fallback TF-IDF: {fallback_e}")
    
    def load_existing_assignments(self):
        """Load existing assignments for comparison"""
        try:
            for assignment_file in self.assignments_dir.glob("*.json"):
                try:
                    with open(assignment_file, 'r', encoding='utf-8') as f:
                        assignment_data = json.load(f)
                        assignment_id = assignment_file.stem
                        self.existing_assignments[assignment_id] = assignment_data
                        
                        # Generate hash for quick comparison
                        if 'content' in assignment_data:
                            content_hash = self._generate_content_hash(assignment_data['content'])
                            self.assignment_hashes[content_hash] = assignment_id
                            
                except Exception as e:
                    logger.warning(f"Failed to load assignment {assignment_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to load existing assignments: {e}")
    
    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from various file formats"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == '.pdf':
                return self._extract_text_from_pdf(file_path)
            elif file_extension == '.docx':
                return self._extract_text_from_docx(file_path)
            elif file_extension == '.txt':
                return self._extract_text_from_txt(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            logger.error(f"Failed to extract text from {file_path}: {e}")
            raise
    
    def _extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF using multiple methods"""
        text = ""
        
        # Try pdfplumber first
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            logger.warning(f"pdfplumber failed, trying PyPDF2: {e}")
        
        # Fallback to PyPDF2
        if not text.strip():
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as e:
                logger.error(f"PyPDF2 also failed: {e}")
        
        return text.strip()
    
    def _extract_text_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Failed to extract text from DOCX: {e}")
            raise
    
    def _extract_text_from_txt(self, file_path: Path) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Failed to extract text from TXT: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Tokenize and remove stop words
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # Lemmatize tokens
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate hash for content"""
        processed_content = self.preprocess_text(content)
        return hashlib.md5(processed_content.encode()).hexdigest()
    
    def _generate_ngram_fingerprints(self, text: str, n: int = 3) -> Set[str]:
        """Generate n-gram fingerprints for text"""
        words = word_tokenize(text.lower())
        ngrams_list = list(ngrams(words, n))
        return {''.join(ngram) for ngram in ngrams_list}
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using TF-IDF and cosine similarity"""
        try:
            if not text1.strip() or not text2.strip():
                return 0.0
            
            # Preprocess texts
            processed_text1 = self.preprocess_text(text1)
            processed_text2 = self.preprocess_text(text2)
            
            # Vectorize texts
            vectors = self.tfidf_vectorizer.fit_transform([processed_text1, processed_text2])
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(vectors)
            return float(similarity_matrix[0, 1])
            
        except Exception as e:
            logger.error(f"Failed to calculate text similarity: {e}")
            return 0.0
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using transformer models"""
        try:
            if not ADVANCED_NLP_AVAILABLE or not self.sentence_transformer:
                # Fallback to basic text similarity if advanced NLP not available
                return self.calculate_text_similarity(text1, text2) * 0.8
            
            # Encode texts
            embeddings1 = self.sentence_transformer.encode([text1])
            embeddings2 = self.sentence_transformer.encode([text2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(embeddings1, embeddings2)[0, 0]
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to calculate semantic similarity: {e}")
            # Fallback to basic text similarity
            return self.calculate_text_similarity(text1, text2) * 0.8
    
    def calculate_fingerprint_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity based on n-gram fingerprints"""
        try:
            if not text1.strip() or not text2.strip():
                return 0.0
            
            # Generate fingerprints
            fingerprints1 = self._generate_ngram_fingerprints(text1, n=3)
            fingerprints2 = self._generate_ngram_fingerprints(text2, n=3)
            
            if not fingerprints1 or not fingerprints2:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = len(fingerprints1.intersection(fingerprints2))
            union = len(fingerprints1.union(fingerprints2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate fingerprint similarity: {e}")
            return 0.0
    
    def detect_plagiarism(self, new_assignment_path: str, student_id: str, assignment_title: str = "") -> Dict:
        """
        Main plagiarism detection function
        
        Returns:
            Dict containing plagiarism analysis results
        """
        try:
            # Extract text from new assignment
            new_text = self.extract_text_from_file(new_assignment_path)
            if not new_text.strip():
                return {
                    "error": "No text content found in the assignment file",
                    "plagiarism_detected": False,
                    "similarity_score": 0.0,
                    "plagiarism_percentage": 0.0
                }
            
            # Generate hash for new assignment
            new_hash = self._generate_content_hash(new_text)
            
            # Check for exact matches
            if new_hash in self.assignment_hashes:
                matched_id = self.assignment_hashes[new_hash]
                return {
                    "plagiarism_detected": True,
                    "similarity_score": 1.0,
                    "plagiarism_percentage": 100.0,
                    "exact_match": True,
                    "matched_assignment": matched_id,
                    "severity": "CRITICAL",
                    "message": "Exact content match detected with existing assignment"
                }
            
            # Compare with existing assignments
            plagiarism_results = []
            max_similarity = 0.0
            
            for assignment_id, assignment_data in self.existing_assignments.items():
                if assignment_id == student_id:  # Skip self-comparison
                    continue
                
                if 'content' not in assignment_data:
                    continue
                
                existing_text = assignment_data['content']
                
                # Calculate multiple similarity metrics
                text_similarity = self.calculate_text_similarity(new_text, existing_text)
                semantic_similarity = self.calculate_semantic_similarity(new_text, existing_text)
                fingerprint_similarity = self.calculate_fingerprint_similarity(new_text, existing_text)
                
                # Weighted average similarity
                weighted_similarity = (text_similarity * 0.4 + 
                                     semantic_similarity * 0.4 + 
                                     fingerprint_similarity * 0.2)
                
                if weighted_similarity > 0.3:  # Only report significant similarities
                    plagiarism_results.append({
                        "assignment_id": assignment_id,
                        "text_similarity": text_similarity,
                        "semantic_similarity": semantic_similarity,
                        "fingerprint_similarity": fingerprint_similarity,
                        "weighted_similarity": weighted_similarity,
                        "student_name": assignment_data.get('student_name', 'Unknown'),
                        "assignment_title": assignment_data.get('title', 'Unknown')
                    })
                
                max_similarity = max(max_similarity, weighted_similarity)
            
            # Determine plagiarism status
            plagiarism_detected = max_similarity > self.similarity_threshold
            plagiarism_percentage = max_similarity * 100
            
            # Determine severity level
            if max_similarity >= 0.9:
                severity = "CRITICAL"
            elif max_similarity >= 0.7:
                severity = "HIGH"
            elif max_similarity >= 0.5:
                severity = "MEDIUM"
            elif max_similarity >= 0.3:
                severity = "LOW"
            else:
                severity = "NONE"
            
            # Generate detailed report
            report = {
                "plagiarism_detected": plagiarism_detected,
                "similarity_score": max_similarity,
                "plagiarism_percentage": plagiarism_percentage,
                "severity": severity,
                "exact_match": False,
                "total_assignments_compared": len(self.existing_assignments),
                "timestamp": datetime.now().isoformat(),
                "student_id": student_id,
                "assignment_title": assignment_title,
                "file_path": new_assignment_path,
                "detailed_comparisons": sorted(plagiarism_results, 
                                            key=lambda x: x['weighted_similarity'], 
                                            reverse=True)[:5]  # Top 5 matches
            }
            
            # Add alert message
            if plagiarism_detected:
                if severity == "CRITICAL":
                    report["alert_message"] = f"🚨 CRITICAL PLAGIARISM ALERT: {plagiarism_percentage:.1f}% similarity detected!"
                elif severity == "HIGH":
                    report["alert_message"] = f"⚠️ HIGH PLAGIARISM RISK: {plagiarism_percentage:.1f}% similarity detected!"
                elif severity == "MEDIUM":
                    report["alert_message"] = f"⚠️ MEDIUM PLAGIARISM RISK: {plagiarism_percentage:.1f}% similarity detected!"
                else:
                    report["alert_message"] = f"ℹ️ LOW SIMILARITY: {plagiarism_percentage:.1f}% similarity detected"
            else:
                report["alert_message"] = "✅ No significant plagiarism detected"
            
            return report
            
        except Exception as e:
            logger.error(f"Plagiarism detection failed: {e}")
            return {
                "error": str(e),
                "plagiarism_detected": False,
                "similarity_score": 0.0,
                "plagiarism_percentage": 0.0
            }
    
    def save_assignment_for_comparison(self, student_id: str, assignment_title: str, 
                                     content: str, file_path: str = "") -> bool:
        """Save assignment for future plagiarism comparison"""
        try:
            assignment_data = {
                "student_id": student_id,
                "title": assignment_title,
                "content": content,
                "file_path": file_path,
                "timestamp": datetime.now().isoformat(),
                "content_hash": self._generate_content_hash(content)
            }
            
            # Save to assignments directory
            assignment_file = self.assignments_dir / f"{student_id}_{assignment_title}.json"
            with open(assignment_file, 'w', encoding='utf-8') as f:
                json.dump(assignment_data, f, indent=2, ensure_ascii=False)
            
            # Update internal tracking
            assignment_id = f"{student_id}_{assignment_title}"
            self.existing_assignments[assignment_id] = assignment_data
            self.assignment_hashes[assignment_data["content_hash"]] = assignment_id
            
            logger.info(f"Assignment saved for comparison: {assignment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save assignment for comparison: {e}")
            return False
    
    def get_plagiarism_statistics(self) -> Dict:
        """Get overall plagiarism statistics"""
        try:
            total_assignments = len(self.existing_assignments)
            if total_assignments == 0:
                return {"total_assignments": 0}
            
            # Analyze all assignments for potential plagiarism
            plagiarism_counts = {"NONE": 0, "LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
            
            for assignment_id, assignment_data in self.existing_assignments.items():
                if 'content' in assignment_data:
                    # This is a simplified check - in practice you'd want more sophisticated analysis
                    plagiarism_counts["NONE"] += 1  # Placeholder
            
            return {
                "total_assignments": total_assignments,
                "plagiarism_distribution": plagiarism_counts,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get plagiarism statistics: {e}")
            return {"error": str(e)}
    
    def export_plagiarism_report(self, output_path: str = None) -> str:
        """Export comprehensive plagiarism report"""
        try:
            if output_path is None:
                output_path = f"plagiarism_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            report_data = {
                "report_generated": datetime.now().isoformat(),
                "total_assignments": len(self.existing_assignments),
                "assignments": self.existing_assignments,
                "statistics": self.get_plagiarism_statistics()
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Plagiarism report exported to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export plagiarism report: {e}")
            return ""

# Example usage and testing
if __name__ == "__main__":
    # Initialize plagiarism checker
    checker = PlagiarismChecker()
    
    # Example: Check a new assignment
    # result = checker.detect_plagiarism("path/to/new_assignment.pdf", "student123", "Essay Assignment")
    # print(json.dumps(result, indent=2))
