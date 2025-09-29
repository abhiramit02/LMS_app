#!/usr/bin/env python3
"""
Test script to check if plagiarism routes can be imported
"""

try:
    print("Testing plagiarism route import...")
    
    # Test 1: Import the router
    from routes.plagiarism import router
    print("✅ Router imported successfully")
    
    # Test 2: Check if routes are registered
    print(f"Router prefix: {router.prefix}")
    print(f"Router tags: {router.tags}")
    
    # Test 3: Check available routes
    routes = [route for route in router.routes]
    print(f"Available routes: {len(routes)}")
    for route in routes:
        print(f"  - {route.methods} {route.path}")
    
    print("✅ All route tests passed!")
    
except Exception as e:
    print(f"❌ Error importing routes: {e}")
    import traceback
    traceback.print_exc()
