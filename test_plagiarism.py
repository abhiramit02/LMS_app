#!/usr/bin/env python3
"""
Test script for the Plagiarism Checker system
This script demonstrates the core functionality of the plagiarism detection system
"""

import json
from pathlib import Path
from plagiarism_checker import PlagiarismChecker

def create_test_assignments():
    """Create sample assignments for testing"""
    test_dir = Path("test_assignments")
    test_dir.mkdir(exist_ok=True)
    
    # Sample assignment 1 - Original content
    assignment1 = """
    Machine Learning is a subset of artificial intelligence that focuses on the development 
    of computer programs that can access data and use it to learn for themselves. The process 
    of learning begins with observations or data, such as examples, direct experience, or 
    instruction, in order to look for patterns in data and make better decisions in the future.
    
    There are three main types of machine learning: supervised learning, unsupervised learning, 
    and reinforcement learning. Supervised learning uses labeled training data to learn the 
    mapping function from input to output. Unsupervised learning finds hidden patterns in 
    unlabeled data. Reinforcement learning learns by interacting with an environment and 
    receiving rewards or penalties.
    """
    
    # Sample assignment 2 - Similar content (potential plagiarism)
    assignment2 = """
    Machine Learning represents a branch of artificial intelligence that concentrates on creating 
    computer programs capable of accessing data and utilizing it for self-learning. The learning 
    process commences with observations or data, including examples, direct experience, or 
    instruction, to identify patterns in data and improve future decision-making.
    
    Machine learning encompasses three primary categories: supervised learning, unsupervised 
    learning, and reinforcement learning. Supervised learning employs labeled training data 
    to understand the mapping function from input to output. Unsupervised learning discovers 
    hidden patterns in unlabeled data. Reinforcement learning acquires knowledge through 
    environment interaction and receiving rewards or penalties.
    """
    
    # Sample assignment 3 - Completely different content
    assignment3 = """
    Data Science is an interdisciplinary field that uses scientific methods, processes, 
    algorithms, and systems to extract knowledge and insights from structured and unstructured data. 
    It combines statistics, data analysis, machine learning, and related methods to understand 
    and analyze actual phenomena with data.
    
    The field of data science has emerged due to the growth and availability of data, 
    the development of computational power, and the advancement of statistical methods. 
    Data scientists use various tools and techniques including Python, R, SQL, and 
    specialized software for data visualization and analysis.
    """
    
    # Save assignments
    assignments = [
        ("student001", "ML_Assignment_1", assignment1),
        ("student002", "ML_Assignment_2", assignment2),
        ("student003", "Data_Science_Assignment", assignment3)
    ]
    
    for student_id, title, content in assignments:
        assignment_data = {
            "student_id": student_id,
            "title": title,
            "content": content,
            "timestamp": "2024-01-01T00:00:00",
            "content_hash": ""
        }
        
        file_path = test_dir / f"{student_id}_{title}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(assignment_data, f, indent=2, ensure_ascii=False)
    
    print("✅ Test assignments created successfully!")
    return test_dir

def test_plagiarism_detection():
    """Test the plagiarism detection system"""
    print("\n🔍 Testing Plagiarism Detection System...")
    
    # Initialize plagiarism checker with test data
    checker = PlagiarismChecker(assignments_dir="test_assignments")
    
    # Test 1: Check for plagiarism between similar assignments
    print("\n📝 Test 1: Checking similarity between ML assignments...")
    
    # Create a new assignment similar to assignment 2
    new_assignment = """
    Machine Learning is a subset of artificial intelligence that focuses on the development 
    of computer programs that can access data and use it to learn for themselves. The process 
    of learning begins with observations or data, such as examples, direct experience, or 
    instruction, in order to look for patterns in data and make better decisions in the future.
    
    There are three main types of machine learning: supervised learning, unsupervised learning, 
    and reinforcement learning. Supervised learning uses labeled training data to learn the 
    mapping function from input to output. Unsupervised learning finds hidden patterns in 
    unlabeled data. Reinforcement learning learns by interacting with an environment and 
    receiving rewards or penalties.
    """
    
    # Save new assignment for testing
    new_assignment_data = {
        "student_id": "student004",
        "title": "ML_Assignment_New",
        "content": new_assignment,
        "timestamp": "2024-01-02T00:00:00",
        "content_hash": ""
    }
    
    test_dir = Path("test_assignments")
    new_file_path = test_dir / "student004_ML_Assignment_New.json"
    with open(new_file_path, 'w', encoding='utf-8') as f:
        json.dump(new_assignment_data, f, indent=2, ensure_ascii=False)
    
    # Now check for plagiarism
    result = checker.detect_plagiarism(
        str(new_file_path), 
        "student004", 
        "ML_Assignment_New"
    )
    
    print("📊 Plagiarism Check Results:")
    print(f"   Plagiarism Detected: {result.get('plagiarism_detected', False)}")
    print(f"   Similarity Score: {result.get('similarity_score', 0):.3f}")
    print(f"   Plagiarism Percentage: {result.get('plagiarism_percentage', 0):.1f}%")
    print(f"   Severity: {result.get('severity', 'UNKNOWN')}")
    print(f"   Alert Message: {result.get('alert_message', 'No message')}")
    
    if result.get('detailed_comparisons'):
        print("\n📋 Detailed Comparisons:")
        for i, comp in enumerate(result['detailed_comparisons'][:3], 1):
            print(f"   {i}. {comp['assignment_title']} (Student: {comp['student_name']})")
            print(f"      Text Similarity: {comp['text_similarity']*100:.1f}%")
            print(f"      Semantic Similarity: {comp['semantic_similarity']*100:.1f}%")
            print(f"      Overall Similarity: {comp['weighted_similarity']*100:.1f}%")
    
    # Test 2: Check statistics
    print("\n📈 Test 2: Checking system statistics...")
    stats = checker.get_plagiarism_statistics()
    print(f"   Total Assignments: {stats.get('total_assignments', 0)}")
    
    # Test 3: Export report
    print("\n📄 Test 3: Exporting plagiarism report...")
    report_path = checker.export_plagiarism_report()
    if report_path:
        print(f"   Report exported to: {report_path}")
    else:
        print("   Failed to export report")
    
    print("\n✅ Plagiarism detection tests completed!")

def cleanup_test_files():
    """Clean up test files"""
    import shutil
    
    test_dir = Path("test_assignments")
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print("🧹 Test files cleaned up!")

if __name__ == "__main__":
    print("🚀 Starting Plagiarism Checker Test Suite...")
    
    try:
        # Create test assignments
        create_test_assignments()
        
        # Run plagiarism detection tests
        test_plagiarism_detection()
        
        # Clean up
        cleanup_test_files()
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
