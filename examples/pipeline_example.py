"""Example demonstrating the recommendation pipeline."""

import asyncio
import os
from app.pipeline import recommend


async def main():
    """
    Example usage of the recommendation pipeline.
    
    This requires actual API keys to be set in your environment.
    """
    # Check if required environment variables are set
    if not os.getenv("GH_TOKEN"):
        print("⚠️  GH_TOKEN environment variable not set")
        print("Please create a .env file with your GitHub token")
        return
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY environment variable not set")
        print("Please add your OpenAI API key to the .env file")
        return
    
    print("🚀 Starting recommendation pipeline...")
    
    # Example user input
    user_text = """
    I'm a Python developer with 3 years of experience. I love working with FastAPI, 
    Django, and building web APIs. I'm also interested in machine learning and have 
    used scikit-learn and pandas. I'd like to contribute to open source projects 
    that involve web development or data science.
    """
    
    try:
        # Get recommendations
        print("🔍 Analyzing your skills and finding matching issues...")
        result = await recommend(user_text, top_k=5)
        
        print(f"\n✅ Found {len(result.items)} recommended issues:")
        print("=" * 60)
        
        for i, issue in enumerate(result.items, 1):
            print(f"\n{i}. {issue.title}")
            print(f"   Repository: {issue.repo}")
            print(f"   Difficulty: {issue.difficulty}")
            print(f"   Match Score: {issue.score:.2f}")
            print(f"   Summary: {issue.summary}")
            print(f"   URL: {issue.url}")
            print(f"   Labels: {', '.join(issue.labels)}")
            
        if not result.items:
            print("\n🤔 No matching issues found. Try expanding your skills description!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure your API keys are valid and you have internet connection.")


if __name__ == "__main__":
    asyncio.run(main()) 