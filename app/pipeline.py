"""Pipeline for generating open source issue recommendations."""

from typing import List
from app.config import get_settings
from app.schemas import RecommendationResponse, SkillProfile, RawIssue, RankedIssue
from app.github_service import GitHubService
from our_agents import profile_run, search_run, match_run


async def recommend(user_text: str, top_k: int = 10) -> RecommendationResponse:
    """
    Generate personalized open source issue recommendations.
    
    This function orchestrates the complete recommendation pipeline:
    1. Extract skills from user text using ProfileAgent
    2. Search for relevant issues using SearchAgent  
    3. Rank and match issues using MatchAgent
    
    Args:
        user_text: User's description of their skills, interests, or experience
        top_k: Maximum number of recommendations to return (default: 10)
        
    Returns:
        RecommendationResponse containing ranked issues
        
    Raises:
        Exception: If any step in the pipeline fails
    """
    print(f"\n🔧 PIPELINE START - top_k={top_k}")
    
    # Get configuration
    settings = get_settings()
    
    # Initialize GitHub service
    github_service = GitHubService(token=settings.GH_TOKEN)
    
    try:
        # Step 1: Extract user's skill profile
        print(f"\n1️⃣  PROFILE AGENT")
        print(f"INPUT: {repr(user_text)}")
        
        profile: SkillProfile = await profile_run(user_text)
        
        print(f"OUTPUT:")
        print(f"  - Skills: {profile.skills}")
        print(f"  - Embedding vector: length={len(profile.embedding)}, first_3={profile.embedding[:3]}")
        
        # Step 2: Search for issues using extracted skills as keywords
        print(f"\n2️⃣  SEARCH AGENT")
        search_limit = max(50, top_k * 3)  # Get more issues than needed for better matching
        print(f"INPUT:")
        print(f"  - Keywords: {profile.skills}")
        print(f"  - Limit: {search_limit}")
        
        raw_issues: List[RawIssue] = await search_run(
            keywords=profile.skills,
            limit=search_limit,
            svc=github_service
        )
        
        print(f"OUTPUT: {len(raw_issues)} issues found")
        for i, issue in enumerate(raw_issues[:5]):  # Show first 5
            print(f"  {i+1}. {issue.title}")
            print(f"     Repo: {issue.repo}")
            print(f"     Labels: {issue.labels}")
            print(f"     Body preview: {issue.body[:100]}{'...' if len(issue.body) > 100 else ''}")
        
        if len(raw_issues) > 5:
            print(f"  ... and {len(raw_issues) - 5} more issues")
        
        # Step 3: Rank and match issues based on user profile
        print(f"\n3️⃣  MATCH AGENT")
        print(f"INPUT:")
        print(f"  - Profile: {len(profile.skills)} skills, embedding_length={len(profile.embedding)}")
        print(f"  - Issues: {len(raw_issues)} raw issues")
        print(f"  - Top K: {top_k}")
        print(f"  - User text: {repr(user_text)}")
        
        ranked_issues: List[RankedIssue] = await match_run(
            profile=profile,
            issues=raw_issues,
            svc=github_service,
            top_k=top_k,
            user_text=user_text
        )
        
        print(f"OUTPUT: {len(ranked_issues)} ranked issues")
        for i, issue in enumerate(ranked_issues):
            print(f"  #{i+1} (score: {issue.score:.3f}, difficulty: {issue.difficulty})")
            print(f"      Title: {issue.title}")
            print(f"      Repo: {issue.repo}")
            print(f"      URL: {issue.url}")
            print(f"      Labels: {issue.labels}")
            print(f"      Summary: {issue.summary}")
            print(f"      Repo Summary: {issue.repo_summary}")
            print(f"      ---")
        
        # Return the recommendation response
        print(f"✅ Pipeline completed. Found {len(ranked_issues)} recommendations")
        for i, issue in enumerate(ranked_issues):
            print(f"  {i+1}. {issue.title} (score: {issue.score:.3f}, difficulty: {issue.difficulty})")
            print(f"     Repo: {issue.repo}")
            print(f"     Repo Summary: {issue.repo_summary[:80]}...")
        
        return RecommendationResponse(items=ranked_issues)
        
    except Exception as e:
        # Log the error and return empty response
        print(f"\n💥 PIPELINE ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("🔄 Returning empty response due to error")
        return RecommendationResponse(items=[]) 