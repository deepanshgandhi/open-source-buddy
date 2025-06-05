"""Main FastAPI application."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import Settings, get_settings
from app.github_service import GitHubService
from app.pipeline import recommend
from app.schemas import RecommendationResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instances
github_service: GitHubService = None


class RecommendRequest(BaseModel):
    skills: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown."""
    global github_service
    
    print("🚀 Starting up application...")
    
    # Initialize services during startup
    settings = get_settings()
    print(f"📝 Loaded settings - GH_TOKEN exists: {bool(settings.GH_TOKEN)}, OPENAI_API_KEY exists: {bool(settings.OPENAI_API_KEY)}")
    
    github_service = GitHubService(token=settings.GH_TOKEN)
    print("✅ GitHub service initialized")
    
    yield
    
    # Cleanup during shutdown
    print("🛑 Shutting down application...")


# Create FastAPI app with lifespan
app = FastAPI(
    title="Open Source Issue Recommender",
    description="AI-powered recommendation system for GitHub issues based on your skills",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


def get_github_service() -> GitHubService:
    """Dependency to get GitHub service instance."""
    if github_service is None:
        raise HTTPException(status_code=500, detail="GitHub service not initialized")
    return github_service


@app.get("/healthz")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendRequest,
    settings: Settings = Depends(get_settings)
) -> RecommendationResponse:
    """
    Get personalized open source issue recommendations.
    
    Args:
        request: User's skills and interests description
        settings: Application settings
        
    Returns:
        RecommendationResponse with ranked issues
    """
    try:
        print(f"\n🔍 Processing recommendation request")
        print(f"📝 Input skills: {repr(request.skills)}")
        
        # Generate recommendations using the pipeline
        result = await recommend(request.skills, top_k=10)
        
        print(f"✅ Successfully generated {len(result.items)} recommendations")
        return result
        
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        print(f"❌ Request failed: {type(e).__name__}: {e}")
        
        # Return error response
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate recommendations: {str(e)}"
        )


def main() -> None:
    """Main entry point for the CLI script."""
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


if __name__ == "__main__":
    main() 