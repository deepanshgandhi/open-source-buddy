from typing import List
from app.schemas import RawIssue
from app.github_service import GitHubService


class SearchAgent:
    async def run(
        self, 
        keywords: List[str], 
        limit: int, 
        svc: GitHubService
    ) -> List[RawIssue]:
        """
        Search for GitHub issues using the provided keywords.
        
        Args:
            keywords: List of search keywords/terms
            limit: Maximum number of issues to return
            svc: GitHubService instance for API calls
            
        Returns:
            List of RawIssue objects
        """
        if not keywords:
            return []
        
        # Use GitHub service to search for issues
        # Set per_kw to distribute limit across keywords
        per_kw = max(1, limit // len(keywords)) if keywords else limit
        
        issues = await svc.search_issues(
            keywords=keywords,
            per_kw=per_kw,
            max_total=limit
        )
        
        return issues


# Factory function for easy import
async def run(keywords: List[str], limit: int, svc: GitHubService) -> List[RawIssue]:
    """
    Search for GitHub issues using the provided keywords.
    
    Args:
        keywords: List of search keywords/terms
        limit: Maximum number of issues to return
        svc: GitHubService instance for API calls
        
    Returns:
        List of RawIssue objects
    """
    agent = SearchAgent()
    return await agent.run(keywords, limit, svc) 