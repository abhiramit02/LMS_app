#!/usr/bin/env python3
"""
Simple test to check if plagiarism checker can initialize
"""

try:
    print("Testing plagiarism checker initialization...")
    from plagiarism_checker import PlagiarismChecker
    
    print("✅ PlagiarismChecker imported successfully")
    
    # Try to initialize
    checker = PlagiarismChecker()
    print("✅ PlagiarismChecker initialized successfully")
    
    # Check what models are available
    print(f"TF-IDF available: {checker.tfidf_vectorizer is not None}")
    print(f"Sentence transformer available: {checker.sentence_transformer is not None}")
    print(f"Advanced NLP available: {hasattr(checker, 'transformer_model') and checker.transformer_model is not None}")
    
    print("✅ All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
