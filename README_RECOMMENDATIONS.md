# AI-Powered Learning Recommendation System for LMS

## Overview

This is a comprehensive, AI-powered recommendation system for Learning Management Systems (LMS) that combines multiple recommendation algorithms to provide personalized learning suggestions. The system uses collaborative filtering, content-based filtering, and transformer-based contextual understanding to deliver intelligent course and material recommendations.

## 🚀 Features

### Core Recommendation Algorithms

1. **Collaborative Filtering**
   - Matrix factorization using SVD (Singular Value Decomposition)
   - Student similarity-based recommendations
   - Fallback to similarity-based approach if Surprise library unavailable

2. **Content-Based Filtering**
   - TF-IDF vectorization of course content
   - Content similarity analysis
   - Student preference learning from performance history

3. **Transformer-Based Recommendations**
   - BERT/DistilBERT embeddings for contextual understanding
   - Learning pattern analysis
   - Contextual scoring based on student behavior

4. **Hybrid Recommendations**
   - Weighted combination of all three methods
   - Configurable weights for different algorithms
   - Intelligent score normalization and ranking

### Advanced Features

- **Next Course Recommendations**: Prerequisites-based course sequencing
- **Learning Path Generation**: Personalized learning journey mapping
- **Learning Material Recommendations**: Adaptive content suggestions based on performance
- **Student Profiling**: Comprehensive learning behavior analysis
- **Progress Analytics**: Visual performance tracking and insights

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Web Application                  │
├─────────────────────────────────────────────────────────────┤
│  /recommendations/* - API Endpoints                        │
│  /ui/recommendations - Web Dashboard                       │
├─────────────────────────────────────────────────────────────┤
│                Recommendation System Core                   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │Collaborative│ │Content-Based│ │Transformer- │          │
│  │  Filtering  │ │  Filtering  │ │   Based     │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
├─────────────────────────────────────────────────────────────┤
│                    Data Layer                              │
│  - Student Activity Data                                   │
│  - Student Scores                                          │
│  - Course Content                                          │
│  - Question Bank                                           │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

### Key Dependencies

- **FastAPI**: Web framework for API endpoints
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and utilities
- **Transformers**: Hugging Face transformer models (BERT/DistilBERT)
- **Surprise**: Collaborative filtering library
- **NLTK**: Natural language processing
- **Chart.js**: Frontend charting library

## 🚀 Quick Start

### 1. Test the System

Run the test suite to verify everything is working:

```bash
python test_recommendations.py
```

### 2. Start the Server

```bash
uvicorn main:app --reload
```

### 3. Access the Dashboard

Open your browser and navigate to:
- **Main Dashboard**: http://localhost:8000/
- **Recommendations UI**: http://localhost:8000/ui/recommendations
- **API Documentation**: http://localhost:8000/docs

## 📊 API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/recommendations/health` | GET | Service health check |
| `/recommendations/stats` | GET | System statistics |
| `/recommendations/profile/{student_id}` | GET | Student profile data |
| `/recommendations/dashboard/{student_id}` | GET | Complete personalized dashboard |

### Recommendation Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/recommendations/collaborative-filtering/{student_id}` | GET | Collaborative filtering recommendations |
| `/recommendations/content-based/{student_id}` | GET | Content-based recommendations |
| `/recommendations/transformer-based/{student_id}` | GET | Transformer-based recommendations |
| `/recommendations/hybrid/{student_id}` | GET | Hybrid recommendations (all methods) |
| `/recommendations/next-courses/{student_id}` | GET | Next course recommendations |
| `/recommendations/learning-materials/{student_id}/{topic_id}` | GET | Learning material suggestions |

### Utility Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/recommendations/learning-path/{student_id}` | GET | Personalized learning path |
| `/recommendations/progress/{student_id}` | GET | Student progress analytics |
| `/recommendations/reload-data` | POST | Reload data and reinitialize system |
| `/recommendations/save-models` | POST | Save trained models |
| `/recommendations/load-models` | POST | Load previously saved models |

## 🎯 Usage Examples

### Python API Usage

```python
from recommendation_system import LMSRecommendationSystem

# Initialize the system
rec_system = LMSRecommendationSystem()

# Get personalized recommendations for a student
student_id = "student_123"
recommendations = rec_system.hybrid_recommendations(student_id, n_recommendations=10)

# Get next course recommendations
next_courses = rec_system.get_next_course_recommendations(student_id, n_recommendations=5)

# Get personalized dashboard
dashboard = rec_system.get_personalized_dashboard_data(student_id)
```

### HTTP API Usage

```bash
# Get hybrid recommendations
curl "http://localhost:8000/recommendations/hybrid/student_123?n_recommendations=5"

# Get student profile
curl "http://localhost:8000/recommendations/profile/student_123"

# Get personalized dashboard
curl "http://localhost:8000/recommendations/dashboard/student_123"
```

## 🔧 Configuration

### Algorithm Weights

The hybrid recommendation system uses configurable weights for different algorithms:

```python
# Default weights in hybrid_recommendations method
weights = {
    'cf': 0.4,    # Collaborative filtering: 40%
    'cb': 0.35,   # Content-based: 35%
    'tf': 0.25    # Transformer-based: 25%
}
```

### Model Parameters

```python
# SVD parameters for collaborative filtering
algo = SVD(
    n_factors=50,      # Number of latent factors
    n_epochs=20,       # Training epochs
    lr_all=0.005,      # Learning rate
    reg_all=0.02       # Regularization
)

# TF-IDF parameters for content-based filtering
tfidf = TfidfVectorizer(
    max_features=100,   # Maximum features
    stop_words='english' # Remove common words
)
```

## 📈 Performance Optimization

### Caching

- Model embeddings are cached after first computation
- Student profiles are computed on-demand and cached
- Recommendation results can be cached for frequently accessed students

### Scalability

- Efficient matrix operations using NumPy
- Lazy loading of transformer models
- Configurable batch processing for large datasets

### Memory Management

- Automatic cleanup of large embeddings
- Efficient data structures for student-topic matrices
- Optional model persistence to disk

## 🧪 Testing

### Run Tests

```bash
# Test core functionality
python test_recommendations.py

# Test with specific student
python -c "
from recommendation_system import LMSRecommendationSystem
rec = LMSRecommendationSystem()
print(rec.get_student_profile('student_1'))
"
```

### Test Data

The system automatically generates dummy data if real data is not available:
- 100 students with learning activities
- 30 topics with content descriptions
- 200 practice questions
- Realistic performance patterns

## 📁 File Structure

```
LMS/
├── recommendation_system.py          # Core recommendation engine
├── routes/
│   └── recommendations.py           # FastAPI router
├── templates/
│   └── recommendations.html         # Web dashboard
├── test_recommendations.py          # Test suite
├── requirements.txt                 # Dependencies
└── README_RECOMMENDATIONS.md       # This file
```

## 🔍 Troubleshooting

### Common Issues

1. **Transformers not loading**
   - Check internet connection for model download
   - Verify sufficient disk space
   - Check Python version compatibility

2. **Memory issues with large datasets**
   - Reduce `max_features` in TF-IDF
   - Lower `n_factors` in SVD
   - Enable model persistence

3. **Slow recommendations**
   - Enable model caching
   - Reduce number of recommendations
   - Use fallback algorithms

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🚀 Deployment

### Production Considerations

1. **Model Persistence**: Save trained models to disk
2. **Caching**: Implement Redis for recommendation caching
3. **Load Balancing**: Use multiple worker processes
4. **Monitoring**: Add health checks and metrics

### Docker Deployment

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Add type hints where possible
- Include docstrings for all functions
- Write comprehensive tests

## 📚 References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/)
- [Transformers Library](https://huggingface.co/transformers/)
- [Surprise Library](https://surprise.readthedocs.io/)
- [Collaborative Filtering](https://en.wikipedia.org/wiki/Collaborative_filtering)
- [Content-Based Filtering](https://en.wikipedia.org/wiki/Recommender_system#Content-based_filtering)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
1. Check the troubleshooting section
2. Review the test examples
3. Open an issue on GitHub
4. Contact the development team

---

**Built with ❤️ for intelligent learning experiences**
