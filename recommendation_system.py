import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Transformer Models
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available. Using fallback methods.")

# Collaborative Filtering
try:
    from surprise import Dataset, Reader, SVD, KNNBasic, accuracy
    from surprise.model_selection import train_test_split
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False
    print("Warning: Surprise library not available. Using fallback collaborative filtering.")


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in cols:
            return cols[name.lower()]
    # fallback: substring match
    for c in df.columns:
        lc = c.lower()
        for name in candidates:
            if name.lower() in lc:
                return c
    return None

def _read_csv_flexible(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, engine="python", on_bad_lines="skip")
        except Exception as e:
            last_err = e
            continue
    raise last_err

class LMSRecommendationSystem:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.models = {}
        self.vectorizers = {}
        self.scalers = {}
        self.transformer_model = None
        self.transformer_tokenizer = None
        
        # Initialize NLTK
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except:
            pass
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Load data
        self.load_data()
        
        # Initialize transformer model
        if TRANSFORMERS_AVAILABLE:
            self.initialize_transformer_model()
    
    def load_data(self):
        """Load all necessary data files and normalize schemas."""
        try:
            # Load raw files if available
            activity_path = self.data_dir / "student_activity.csv"
            scores_path = self.data_dir / "student_scores.csv"
            content_path = self.data_dir / "student_content.csv"
            qbank_path = self.data_dir / "question_bank.csv"

            self.student_activity = _read_csv_flexible(activity_path) if activity_path.exists() else pd.DataFrame()
            raw_scores = _read_csv_flexible(scores_path) if scores_path.exists() else pd.DataFrame()
            raw_content = _read_csv_flexible(content_path) if content_path.exists() else pd.DataFrame()
            self.question_bank = _read_csv_flexible(qbank_path) if qbank_path.exists() else pd.DataFrame()

            # Normalize scores schema
            if not raw_scores.empty:
                sid_col = _find_col(raw_scores, ["student_id", "sid", "student", "user_id"]) or "student_id"
                topic_col = _find_col(raw_scores, ["topic_id", "topic", "module", "chapter", "unit"]) or "topic"
                score_col = _find_col(raw_scores, ["score"]) or "score"
                max_col = _find_col(raw_scores, ["max_score", "total", "full_score"])
                attempts_col = _find_col(raw_scores, ["attempts", "num_attempts"])  # optional

                scores = pd.DataFrame()
                scores['student_id'] = raw_scores[sid_col].astype(str).str.strip().str.lower()
                scores['topic_id'] = raw_scores[topic_col].astype(str).str.strip().str.lower()
                scores['score'] = pd.to_numeric(raw_scores[score_col], errors='coerce').fillna(0)
                if max_col and max_col in raw_scores.columns:
                    scores['max_score'] = pd.to_numeric(raw_scores[max_col], errors='coerce').fillna(0)
                else:
                    # default if missing
                    scores['max_score'] = 100.0
                scores['attempts'] = pd.to_numeric(raw_scores[attempts_col], errors='coerce').fillna(1) if attempts_col and attempts_col in raw_scores.columns else 1
                self.student_scores = scores
            else:
                self.student_scores = pd.DataFrame(columns=['student_id','topic_id','score','max_score','attempts'])

            # Normalize content schema
            if not raw_content.empty:
                topic_col = _find_col(raw_content, ["topic_id", "topic", "module", "chapter", "unit"]) or "topic"
                title_col = _find_col(raw_content, ["title", "term", "name"]) or _find_col(raw_content, ["topic"]) or topic_col
                desc_col = _find_col(raw_content, ["description", "definition", "summary"]) or title_col
                diff_col = _find_col(raw_content, ["difficulty", "level"])  # optional
                duration_col = _find_col(raw_content, ["duration"])  # optional
                prereq_col = _find_col(raw_content, ["prerequisites", "prereqs"])  # optional

                content = pd.DataFrame()
                content['topic_id'] = raw_content[topic_col].astype(str).str.strip().str.lower()
                content['title'] = raw_content[title_col].astype(str)
                content['description'] = raw_content[desc_col].astype(str)
                content['difficulty'] = raw_content[diff_col].astype(str) if diff_col and diff_col in raw_content.columns else 'Medium'
                content['duration'] = pd.to_numeric(raw_content[duration_col], errors='coerce').fillna(30) if duration_col and duration_col in raw_content.columns else 30
                content['prerequisites'] = raw_content[prereq_col].astype(str) if prereq_col and prereq_col in raw_content.columns else ''
                self.student_content = content
            else:
                self.student_content = pd.DataFrame(columns=['topic_id','title','description','difficulty','duration','prerequisites'])

            # Normalize activity schema (optional)
            if not self.student_activity.empty:
                # best-effort normalization
                sid_col = _find_col(self.student_activity, ["student_id", "sid", "student", "user_id"]) or "student_id"
                time_spent_col = _find_col(self.student_activity, ["time_spent", "minutes", "duration"])  # optional
                completion_col = _find_col(self.student_activity, ["completion_rate", "completion", "progress"])  # optional
                course_col = _find_col(self.student_activity, ["course_id", "course"])  # optional
                topic_col = _find_col(self.student_activity, ["topic_id", "topic"])  # optional
                ts_col = _find_col(self.student_activity, ["timestamp", "time", "date"])  # optional

                act = pd.DataFrame()
                act['student_id'] = self.student_activity[sid_col].astype(str).str.strip().str.lower()
                if time_spent_col and time_spent_col in self.student_activity.columns:
                    act['time_spent'] = pd.to_numeric(self.student_activity[time_spent_col], errors='coerce').fillna(0)
                else:
                    act['time_spent'] = 0
                if completion_col and completion_col in self.student_activity.columns:
                    comp = pd.to_numeric(self.student_activity[completion_col], errors='coerce')
                    # if looks like percent >1, scale down
                    act['completion_rate'] = np.where(comp > 1, comp / 100.0, comp).fillna(0)
                else:
                    act['completion_rate'] = 0
                act['course_id'] = self.student_activity[course_col].astype(str) if course_col and course_col in self.student_activity.columns else ''
                act['topic_id'] = self.student_activity[topic_col].astype(str).str.strip().str.lower() if topic_col and topic_col in self.student_activity.columns else ''
                if ts_col and ts_col in self.student_activity.columns:
                    act['timestamp'] = pd.to_datetime(self.student_activity[ts_col], errors='coerce')
                else:
                    act['timestamp'] = pd.NaT
                self.student_activity = act

            print(f"Loaded data -> scores:{len(self.student_scores)} content:{len(self.student_content)} activity:{len(self.student_activity)}")
        except Exception as e:
            print(f"Error loading data: {e}")
            # Create dummy data for testing
            self.create_dummy_data()
    
    def create_dummy_data(self):
        """Create dummy data for testing if real data is not available"""
        np.random.seed(42)
        
        # Dummy student activity
        students = [f"student_{i}" for i in range(1, 101)]
        courses = [f"course_{i}" for i in range(1, 21)]
        topics = [f"topic_{i}" for i in range(1, 31)]
        
        self.student_activity = pd.DataFrame({
            'student_id': np.random.choice(students, 1000),
            'course_id': np.random.choice(courses, 1000),
            'topic_id': np.random.choice(topics, 1000),
            'time_spent': np.random.exponential(30, 1000),
            'completion_rate': np.random.beta(2, 2, 1000),
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='H')
        })
        
        # Dummy student scores
        self.student_scores = pd.DataFrame({
            'student_id': np.random.choice(students, 2000),
            'topic_id': np.random.choice(topics, 2000),
            'score': np.random.normal(75, 15, 2000),
            'max_score': 100,
            'attempts': np.random.poisson(2, 2000)
        })
        
        # Dummy content
        self.student_content = pd.DataFrame({
            'topic_id': topics,
            'title': [f"Introduction to {topic}" for topic in topics],
            'description': [f"This module covers the fundamentals of {topic}" for topic in topics],
            'difficulty': np.random.choice(['Beginner', 'Intermediate', 'Advanced'], 30),
            'duration': np.random.exponential(45, 30),
            'prerequisites': [f"topic_{max(1, i-1)}" for i in range(1, 31)]
        })
        
        # Dummy question bank
        self.question_bank = pd.DataFrame({
            'question_id': [f"q_{i}" for i in range(1, 201)],
            'topic_id': np.random.choice(topics, 200),
            'question_text': [f"Sample question {i} about topic" for i in range(1, 201)],
            'difficulty': np.random.choice(['Easy', 'Medium', 'Hard'], 200),
            'question_type': np.random.choice(['MCQ', 'Essay', 'Programming'], 200)
        })
    
    def initialize_transformer_model(self):
        """Initialize BERT model for contextual understanding"""
        try:
            model_name = "distilbert-base-uncased"  # Faster alternative to BERT
            self.transformer_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.transformer_model = AutoModel.from_pretrained(model_name)
            print(f"Loaded transformer model: {model_name}")
        except Exception as e:
            print(f"Error loading transformer model: {e}")
            self.transformer_model = None
            self.transformer_tokenizer = None
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in self.stop_words]
        return " ".join(tokens)
    
    def get_student_profile(self, student_id: str) -> Dict:
        """Get comprehensive student profile"""
        # Normalize student_id to lowercase for consistent matching
        student_id_normalized = student_id.lower()
        
        student_activity = self.student_activity[self.student_activity['student_id'] == student_id_normalized]
        student_scores = self.student_scores[self.student_scores['student_id'] == student_id_normalized]
        
        # Create a basic profile even if no data exists
        profile = {
            'student_id': student_id,
            'total_time_spent': float(student_activity['time_spent'].sum() if not student_activity.empty else 0),
            'avg_completion_rate': float(student_activity['completion_rate'].mean() if not student_activity.empty else 0),
            'total_attempts': int(student_scores['attempts'].sum() if not student_scores.empty else 0),
            'avg_score': float(student_scores['score'].mean() if not student_scores.empty else 0),
            'topics_attempted': int(student_scores['topic_id'].nunique() if not student_scores.empty else 0),
            'courses_enrolled': int(student_activity['course_id'].nunique() if not student_activity.empty else 0)
        }
        
        # Learning patterns
        if not student_activity.empty and 'timestamp' in student_activity.columns:
            try:
                profile['preferred_time'] = int(student_activity['timestamp'].dt.hour.mode().iloc[0] if not student_activity['timestamp'].dt.hour.mode().empty else 12)
                profile['avg_session_duration'] = float(student_activity['time_spent'].mean())
            except:
                profile['preferred_time'] = 12
                profile['avg_session_duration'] = 0.0
        else:
            profile['preferred_time'] = 12
            profile['avg_session_duration'] = 0.0
        
        return profile
    
    def collaborative_filtering_recommendations(self, student_id: str, n_recommendations: int = 5) -> List[Dict]:
        """Generate recommendations using collaborative filtering"""
        if not SURPRISE_AVAILABLE:
            return self.fallback_collaborative_filtering(student_id, n_recommendations)
        
        try:
            # Normalize student_id for consistent matching
            student_id_normalized = student_id.lower()
            
            # Prepare data for Surprise
            reader = Reader(rating_scale=(0, 100))
            data = Dataset.load_from_df(
                self.student_scores[['student_id', 'topic_id', 'score']], 
                reader
            )
            
            # Train SVD model
            trainset = data.build_full_trainset()
            algo = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02)
            algo.fit(trainset)
            
            # Get all topics
            all_topics = self.student_content['topic_id'].unique()
            
            # Predict scores for student on all topics
            predictions = []
            for topic in all_topics:
                pred = algo.predict(student_id_normalized, topic)
                predictions.append({
                    'topic_id': topic,
                    'predicted_score': pred.est,
                    'confidence': 1.0 - abs(pred.est - pred.actual) / 100 if pred.actual is not None else 0.5,
                    'method': 'collaborative_filtering'
                })
            
            # Sort by predicted score and return top recommendations
            predictions.sort(key=lambda x: x['predicted_score'], reverse=True)
            
            # Filter out topics student has already attempted
            attempted_topics = set(self.student_scores[self.student_scores['student_id'] == student_id_normalized]['topic_id'])
            recommendations = [p for p in predictions if p['topic_id'] not in attempted_topics]
            
            return recommendations[:n_recommendations]
            
        except Exception as e:
            print(f"Error in collaborative filtering: {e}")
            return self.fallback_collaborative_filtering(student_id, n_recommendations)
    
    def fallback_collaborative_filtering(self, student_id: str, n_recommendations: int = 5) -> List[Dict]:
        """Fallback collaborative filtering using simple similarity"""
        try:
            # Normalize student_id for consistent matching
            student_id_normalized = student_id.lower()
            
            # Create student-topic matrix
            student_topic_matrix = self.student_scores.pivot_table(
                index='student_id', 
                columns='topic_id', 
                values='score', 
                aggfunc='mean'
            ).fillna(0)
            
            if student_id_normalized not in student_topic_matrix.index:
                return []
            
            # Calculate similarity with other students
            student_similarities = cosine_similarity(
                student_topic_matrix.loc[[student_id_normalized]], 
                student_topic_matrix
            )[0]
            
            # Find most similar students
            similar_students = np.argsort(student_similarities)[::-1][1:6]  # Top 5 similar students
            
            # Get topics that similar students performed well in
            recommendations = []
            for sim_student_idx in similar_students:
                sim_student_id = student_topic_matrix.index[sim_student_idx]
                sim_student_topics = self.student_scores[
                    (self.student_scores['student_id'] == sim_student_id) & 
                    (self.student_scores['score'] >= 80)
                ]['topic_id'].unique()
                
                for topic in sim_student_topics:
                    if topic not in [r['topic_id'] for r in recommendations]:
                        recommendations.append({
                            'topic_id': topic,
                            'predicted_score': 85,  # Estimated based on similar students
                            'confidence': student_similarities[sim_student_idx],
                            'method': 'similarity_based'
                        })
            
            return recommendations[:n_recommendations]
            
        except Exception as e:
            print(f"Error in fallback collaborative filtering: {e}")
            return []
    
    def content_based_filtering(self, student_id: str, n_recommendations: int = 5) -> List[Dict]:
        """Generate recommendations using content-based filtering"""
        try:
            # Normalize student_id for consistent matching
            student_id_normalized = student_id.lower()
            
            # Get student's learning history
            student_scores = self.student_scores[self.student_scores['student_id'] == student_id_normalized]
            
            if student_scores.empty:
                return []
            
            # Create student preference vector
            student_preferences = {}
            for _, row in student_scores.iterrows():
                topic = row['topic_id']
                score = row['score']
                if topic not in student_preferences:
                    student_preferences[topic] = []
                student_preferences[topic].append(score)
            
            # Average scores per topic
            avg_preferences = {topic: np.mean(scores) for topic, scores in student_preferences.items()}
            
            # Get content features
            content_features = self.student_content.copy()
            content_features['description_processed'] = content_features['description'].apply(self.preprocess_text)
            
            # TF-IDF vectorization
            tfidf = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf_matrix = tfidf.fit_transform(content_features['description_processed'])
            
            # Calculate content similarity
            content_similarity = cosine_similarity(tfidf_matrix)
            
            # Find topics similar to student's preferred topics
            recommendations = []
            for preferred_topic, preference_score in avg_preferences.items():
                if preferred_topic in content_features['topic_id'].values:
                    topic_idx = content_features[content_features['topic_id'] == preferred_topic].index[0]
                    similar_topics = content_similarity[topic_idx]
                    
                    # Get top similar topics
                    similar_indices = np.argsort(similar_topics)[::-1][1:6]
                    
                    for idx in similar_indices:
                        topic_id = content_features.iloc[idx]['topic_id']
                        if topic_id not in [r['topic_id'] for r in recommendations]:
                            recommendations.append({
                                'topic_id': topic_id,
                                'similarity_score': similar_topics[idx],
                                'predicted_score': preference_score * similar_topics[idx] / 100 * 100,
                                'confidence': similar_topics[idx],
                                'method': 'content_based'
                            })
            
            # Sort by predicted score and remove duplicates
            recommendations = sorted(recommendations, key=lambda x: x['predicted_score'], reverse=True)
            unique_recommendations = []
            seen_topics = set()
            for rec in recommendations:
                if rec['topic_id'] not in seen_topics:
                    unique_recommendations.append(rec)
                    seen_topics.add(rec['topic_id'])
            
            return unique_recommendations[:n_recommendations]
            
        except Exception as e:
            print(f"Error in content-based filtering: {e}")
            return []
    
    def transformer_based_recommendations(self, student_id: str, n_recommendations: int = 5) -> List[Dict]:
        """Generate recommendations using transformer models for contextual understanding"""
        if not TRANSFORMERS_AVAILABLE or self.transformer_model is None:
            return []
        
        try:
            # Get student's learning context
            student_profile = self.get_student_profile(student_id)
            student_scores = self.student_scores[self.student_scores['student_id'] == student_id]
            
            if student_scores.empty:
                return []
            
            # Analyze student's learning patterns
            learning_context = self.analyze_learning_context(student_scores, student_profile)
            
            # Get content embeddings
            content_embeddings = self.get_content_embeddings()
            
            # Calculate contextual similarity
            recommendations = []
            for topic_id, embedding in content_embeddings.items():
                if topic_id not in student_scores['topic_id'].values:
                    contextual_score = self.calculate_contextual_score(
                        embedding, learning_context, student_profile
                    )
                    
                    recommendations.append({
                        'topic_id': topic_id,
                        'contextual_score': contextual_score,
                        'predicted_score': contextual_score * 100,
                        'confidence': min(contextual_score * 1.2, 1.0),
                        'method': 'transformer_based'
                    })
            
            # Sort by contextual score
            recommendations.sort(key=lambda x: x['contextual_score'], reverse=True)
            return recommendations[:n_recommendations]
            
        except Exception as e:
            print(f"Error in transformer-based recommendations: {e}")
            return []
    
    def analyze_learning_context(self, student_scores: pd.DataFrame, student_profile: Dict) -> Dict:
        """Analyze student's learning context and patterns"""
        context = {
            'strength_areas': [],
            'weakness_areas': [],
            'learning_pace': 'medium',
            'preferred_difficulty': 'medium',
            'engagement_pattern': 'consistent'
        }
        
        if not student_scores.empty:
            # Analyze strength and weakness areas
            topic_performance = student_scores.groupby('topic_id')['score'].agg(['mean', 'count']).reset_index()
            
            strong_topics = topic_performance[topic_performance['mean'] >= 80]['topic_id'].tolist()
            weak_topics = topic_performance[topic_performance['mean'] < 60]['topic_id'].tolist()
            
            context['strength_areas'] = strong_topics
            context['weakness_areas'] = weak_topics
            
            # Analyze learning pace
            avg_score = student_scores['score'].mean()
            if avg_score >= 85:
                context['learning_pace'] = 'fast'
            elif avg_score <= 65:
                context['learning_pace'] = 'slow'
            
            # Analyze preferred difficulty
            if student_profile.get('avg_completion_rate', 0) >= 0.8:
                context['preferred_difficulty'] = 'easy'
            elif student_profile.get('avg_completion_rate', 0) <= 0.4:
                context['preferred_difficulty'] = 'hard'
        
        return context
    
    def get_content_embeddings(self) -> Dict[str, np.ndarray]:
        """Get embeddings for all content using transformer model"""
        embeddings = {}
        
        try:
            for _, row in self.student_content.iterrows():
                topic_id = row['topic_id']
                text = f"{row['title']} {row['description']}"
                
                # Tokenize and get embeddings
                inputs = self.transformer_tokenizer(
                    text, 
                    return_tensors="pt", 
                    max_length=512, 
                    truncation=True, 
                    padding=True
                )
                
                with torch.no_grad():
                    outputs = self.transformer_model(**inputs)
                    # Use [CLS] token embedding
                    embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
                    embeddings[topic_id] = embedding
                    
        except Exception as e:
            print(f"Error getting content embeddings: {e}")
        
        return embeddings
    
    def calculate_contextual_score(self, content_embedding: np.ndarray, learning_context: Dict, student_profile: Dict) -> float:
        """Calculate contextual score based on learning context and content"""
        score = 0.5  # Base score
        
        # Adjust based on learning pace
        if learning_context['learning_pace'] == 'fast':
            score += 0.1
        elif learning_context['learning_pace'] == 'slow':
            score -= 0.1
        
        # Adjust based on completion rate
        completion_rate = student_profile.get('avg_completion_rate', 0.5)
        score += (completion_rate - 0.5) * 0.2
        
        # Adjust based on engagement
        if student_profile.get('total_time_spent', 0) > 1000:  # High engagement
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def hybrid_recommendations(self, student_id: str, n_recommendations: int = 10) -> List[Dict]:
        """Generate hybrid recommendations combining all methods"""
        try:
            # Get recommendations from all methods
            cf_recs = self.collaborative_filtering_recommendations(student_id, n_recommendations)
            cb_recs = self.content_based_filtering(student_id, n_recommendations)
            tf_recs = self.transformer_based_recommendations(student_id, n_recommendations)
            
            # Combine and score recommendations
            all_recommendations = {}
            
            # Process collaborative filtering recommendations
            for rec in cf_recs:
                topic_id = rec['topic_id']
                if topic_id not in all_recommendations:
                    all_recommendations[topic_id] = {
                        'topic_id': topic_id,
                        'cf_score': rec.get('predicted_score', 0),
                        'cb_score': 0,
                        'tf_score': 0,
                        'final_score': 0,
                        'methods': []
                    }
                all_recommendations[topic_id]['cf_score'] = rec.get('predicted_score', 0)
                all_recommendations[topic_id]['methods'].append('collaborative_filtering')
            
            # Process content-based recommendations
            for rec in cb_recs:
                topic_id = rec['topic_id']
                if topic_id not in all_recommendations:
                    all_recommendations[topic_id] = {
                        'topic_id': topic_id,
                        'cf_score': 0,
                        'cb_score': 0,
                        'tf_score': 0,
                        'final_score': 0,
                        'methods': []
                    }
                all_recommendations[topic_id]['cb_score'] = rec.get('predicted_score', 0)
                all_recommendations[topic_id]['methods'].append('content_based')
            
            # Process transformer-based recommendations
            for rec in tf_recs:
                topic_id = rec['topic_id']
                if topic_id not in all_recommendations:
                    all_recommendations[topic_id] = {
                        'topic_id': topic_id,
                        'cf_score': 0,
                        'tf_score': 0,
                        'cb_score': 0,
                        'final_score': 0,
                        'methods': []
                    }
                all_recommendations[topic_id]['tf_score'] = rec.get('predicted_score', 0)
                all_recommendations[topic_id]['methods'].append('transformer_based')
            
            # Calculate final scores using weighted combination
            for topic_id, rec in all_recommendations.items():
                # Normalize scores to 0-100 range
                cf_norm = rec['cf_score'] / 100 if rec['cf_score'] > 0 else 0
                cb_norm = rec['cb_score'] / 100 if rec['cb_score'] > 0 else 0
                tf_norm = rec['tf_score'] / 100 if rec['tf_score'] > 0 else 0
                
                # Weighted combination (can be adjusted based on performance)
                weights = {'cf': 0.4, 'cb': 0.35, 'tf': 0.25}
                final_score = (
                    cf_norm * weights['cf'] + 
                    cb_norm * weights['cb'] + 
                    tf_norm * weights['tf']
                )
                
                rec['final_score'] = final_score * 100
                
                # Add required fields for Pydantic validation
                rec['predicted_score'] = rec['final_score']
                rec['confidence'] = min(final_score * 1.2, 1.0)  # Convert to 0-1 range
                rec['method'] = 'hybrid'
            
            # Sort by final score and return top recommendations
            sorted_recommendations = sorted(
                all_recommendations.values(), 
                key=lambda x: x['final_score'], 
                reverse=True
            )
            
            return sorted_recommendations[:n_recommendations]
            
        except Exception as e:
            print(f"Error in hybrid recommendations: {e}")
            return []
    
    def get_next_course_recommendations(self, student_id: str, n_recommendations: int = 5) -> List[Dict]:
        """Get next course/module recommendations based on student progress"""
        try:
            # Get student's current progress
            student_scores = self.student_scores[self.student_scores['student_id'] == student_id]
            student_activity = self.student_activity[self.student_activity['student_id'] == student_id]
            
            if student_scores.empty:
                return []
            
            # Find completed topics (score >= 70)
            completed_topics = set(
                student_scores[student_scores['score'] >= 70]['topic_id']
            )
            
            # Find topics in progress (score < 70 but > 0)
            in_progress_topics = set(
                student_scores[(student_scores['score'] < 70) & (student_scores['score'] > 0)]['topic_id']
            )
            
            # Get all available topics
            all_topics = set(self.student_content['topic_id'].unique())
            
            # Find next logical topics based on prerequisites
            next_topics = []
            for _, content_row in self.student_content.iterrows():
                topic_id = content_row['topic_id']
                prerequisites = str(content_row['prerequisites']).split(',')
                
                # Check if prerequisites are met
                prereqs_met = all(
                    prereq.strip() in completed_topics 
                    for prereq in prerequisites 
                    if prereq.strip() and prereq.strip() != 'nan'
                )
                
                if prereqs_met and topic_id not in completed_topics and topic_id not in in_progress_topics:
                    # Calculate readiness score
                    readiness_score = self.calculate_readiness_score(
                        topic_id, completed_topics, student_scores
                    )
                    
                    next_topics.append({
                        'topic_id': topic_id,
                        'title': content_row['title'],
                        'description': content_row['description'],
                        'difficulty': content_row['difficulty'],
                        'duration': content_row['duration'],
                        'readiness_score': readiness_score,
                        'prerequisites': prerequisites
                    })
            
            # Sort by readiness score and return top recommendations
            next_topics.sort(key=lambda x: x['readiness_score'], reverse=True)
            return next_topics[:n_recommendations]
            
        except Exception as e:
            print(f"Error getting next course recommendations: {e}")
            return []
    
    def calculate_readiness_score(self, topic_id: str, completed_topics: set, student_scores: pd.DataFrame) -> float:
        """Calculate how ready a student is for a specific topic"""
        score = 0.5  # Base score
        
        # Bonus for having many completed topics (general knowledge)
        score += min(len(completed_topics) * 0.02, 0.3)
        
        # Bonus for high performance in completed topics
        if not student_scores.empty:
            avg_completed_score = student_scores[
                student_scores['topic_id'].isin(completed_topics)
            ]['score'].mean()
            if not pd.isna(avg_completed_score):
                score += (avg_completed_score - 70) * 0.002  # Small bonus for high scores
        
        return max(0.0, min(1.0, score))
    
    def get_learning_material_recommendations(self, student_id: str, topic_id: str, n_recommendations: int = 5) -> List[Dict]:
        """Get additional learning material recommendations for a specific topic"""
        try:
            # Get questions related to the topic
            topic_questions = self.question_bank[self.question_bank['topic_id'] == topic_id]
            
            if topic_questions.empty:
                return []
            
            # Get student's performance on this topic
            student_topic_scores = self.student_scores[
                (self.student_scores['student_id'] == student_id) & 
                (self.student_scores['topic_id'] == topic_id)
            ]
            
            avg_score = student_topic_scores['score'].mean() if not student_topic_scores.empty else 50
            
            # Recommend materials based on performance
            recommendations = []
            
            if avg_score < 60:  # Need remedial materials
                # Recommend easier questions and additional practice
                easy_questions = topic_questions[topic_questions['difficulty'] == 'Easy']
                if not easy_questions.empty:
                    recommendations.extend([
                        {
                            'type': 'practice_question',
                            'content': q['question_text'],
                            'difficulty': q['difficulty'],
                            'reason': 'Remedial practice needed'
                        }
                        for _, q in easy_questions.head(3).iterrows()
                    ])
                
                recommendations.append({
                    'type': 'additional_resource',
                    'content': f"Review materials for {topic_id}",
                    'difficulty': 'Beginner',
                    'reason': 'Foundation reinforcement needed'
                })
            
            elif avg_score >= 80:  # Ready for advanced materials
                # Recommend challenging questions and advanced topics
                hard_questions = topic_questions[topic_questions['difficulty'] == 'Hard']
                if not hard_questions.empty:
                    recommendations.extend([
                        {
                            'type': 'challenge_question',
                            'content': q['question_text'],
                            'difficulty': q['difficulty'],
                            'reason': 'Advanced challenge'
                        }
                        for _, q in hard_questions.head(3).iterrows()
                    ])
                
                recommendations.append({
                    'type': 'advanced_resource',
                    'content': f"Advanced concepts in {topic_id}",
                    'difficulty': 'Advanced',
                    'reason': 'Ready for advanced material'
                })
            
            else:  # Moderate performance
                # Recommend balanced materials
                medium_questions = topic_questions[topic_questions['difficulty'] == 'Medium']
                if not medium_questions.empty:
                    recommendations.extend([
                        {
                            'type': 'practice_question',
                            'content': q['question_text'],
                            'difficulty': q['difficulty'],
                            'reason': 'Balanced practice'
                        }
                        for _, q in medium_questions.head(2).iterrows()
                    ])
            
            return recommendations[:n_recommendations]
            
        except Exception as e:
            print(f"Error getting learning material recommendations: {e}")
            return []
    
    def get_personalized_dashboard_data(self, student_id: str) -> Dict:
        """Get personalized dashboard data for a student"""
        try:
            print(f"DEBUG: Starting dashboard data generation for {student_id}")
            
            # Normalize student_id for consistent matching
            student_id_normalized = student_id.lower()
            
            profile = self.get_student_profile(student_id)
            print(f"DEBUG: Profile generated: {profile}")
            
            # Get recommendations with error handling
            try:
                hybrid_recs = self.hybrid_recommendations(student_id, 5)
                print(f"DEBUG: Hybrid recommendations: {len(hybrid_recs)} items")
            except Exception as e:
                print(f"DEBUG: Error in hybrid recommendations: {e}")
                hybrid_recs = []
            
            try:
                next_courses = self.get_next_course_recommendations(student_id, 3)
                print(f"DEBUG: Next courses: {len(next_courses)} items")
            except Exception as e:
                print(f"DEBUG: Error in next courses: {e}")
                next_courses = []
            
            # Get learning progress
            student_scores = self.student_scores[self.student_scores['student_id'] == student_id_normalized]
            progress_data = {}
            
            if not student_scores.empty:
                for topic_id in student_scores['topic_id'].unique():
                    topic_scores = student_scores[student_scores['topic_id'] == topic_id]
                    progress_data[topic_id] = {
                        'avg_score': float(topic_scores['score'].mean()),
                        'attempts': int(topic_scores['attempts'].sum()),
                        'last_attempt': float(topic_scores['score'].iloc[-1] if len(topic_scores) > 0 else 0)
                    }
            
            print(f"DEBUG: Progress data: {progress_data}")
            
            # Generate learning path with error handling
            try:
                learning_path = self.generate_learning_path(student_id)
                print(f"DEBUG: Learning path: {len(learning_path)} items")
            except Exception as e:
                print(f"DEBUG: Error in learning path: {e}")
                learning_path = []
            
            dashboard_data = {
                'profile': profile,
                'recommendations': hybrid_recs,
                'next_courses': next_courses,
                'progress': progress_data,
                'strengths': [topic for topic, data in progress_data.items() if data['avg_score'] >= 80],
                'weaknesses': [topic for topic, data in progress_data.items() if data['avg_score'] < 60],
                'learning_path': learning_path
            }
            
            print(f"DEBUG: Final dashboard data keys: {list(dashboard_data.keys())}")
            return dashboard_data
            
        except Exception as e:
            print(f"Error getting personalized dashboard data: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def generate_learning_path(self, student_id: str) -> List[Dict]:
        """Generate a personalized learning path for the student"""
        try:
            # Normalize student_id for consistent matching
            student_id_normalized = student_id.lower()
            
            # Get current progress and recommendations
            next_courses = self.get_next_course_recommendations(student_id, 10)
            student_scores = self.student_scores[self.student_scores['student_id'] == student_id_normalized]
            
            # Create learning path
            learning_path = []
            
            # Add completed topics
            completed_topics = student_scores[student_scores['score'] >= 70]['topic_id'].unique()
            for topic in completed_topics:
                learning_path.append({
                    'topic_id': topic,
                    'status': 'completed',
                    'score': student_scores[student_scores['topic_id'] == topic]['score'].max(),
                    'order': len(learning_path)
                })
            
            # Add recommended next topics
            for i, course in enumerate(next_courses):
                learning_path.append({
                    'topic_id': course['topic_id'],
                    'status': 'recommended',
                    'readiness_score': course['readiness_score'],
                    'difficulty': course['difficulty'],
                    'order': len(learning_path)
                })
            
            return learning_path
            
        except Exception as e:
            print(f"Error generating learning path: {e}")
            return []
    
    def save_models(self, models_dir: str = "models"):
        """Save trained models for later use"""
        try:
            models_path = Path(models_dir)
            models_path.mkdir(exist_ok=True)
            
            # Save recommendation models
            models_data = {
                'vectorizers': self.vectorizers,
                'scalers': self.scalers,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(models_path / "recommendation_models.pkl", "wb") as f:
                pickle.dump(models_data, f)
            
            print(f"Models saved to {models_path}")
            
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def load_models(self, models_dir: str = "models"):
        """Load previously saved models"""
        try:
            models_path = Path(models_dir) / "recommendation_models.pkl"
            
            if models_path.exists():
                with open(models_path, "rb") as f:
                    models_data = pickle.load(f)
                
                self.vectorizers = models_data.get('vectorizers', {})
                self.scalers = models_data.get('scalers', {})
                
                print(f"Models loaded from {models_path}")
                
        except Exception as e:
            print(f"Error loading models: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Initialize recommendation system
    rec_system = LMSRecommendationSystem()
    
    # Test with a sample student
    test_student = "student_1"
    
    print("=== Testing Recommendation System ===")
    print(f"Student: {test_student}")
    
    # Get personalized dashboard
    dashboard = rec_system.get_personalized_dashboard_data(test_student)
    print(f"\nDashboard data keys: {list(dashboard.keys())}")
    
    # Get hybrid recommendations
    recommendations = rec_system.hybrid_recommendations(test_student, 5)
    print(f"\nTop 5 recommendations:")
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"{i}. {rec['topic_id']} - Score: {rec['final_score']:.1f}")
    
    # Get next course recommendations
    next_courses = rec_system.get_next_course_recommendations(test_student, 3)
    print(f"\nNext course recommendations:")
    for i, course in enumerate(next_courses[:3], 1):
        print(f"{i}. {course['title']} - Readiness: {course['readiness_score']:.2f}")
    
    # Save models
    rec_system.save_models()
