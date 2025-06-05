import asyncio
from typing import List
from github import Github
from github.Issue import Issue

from app.schemas import RawIssue


class GitHubService:
    def __init__(self, token: str):
        """Initialize GitHub service with authentication token."""
        self.client = Github(token, per_page=50)
    
    async def search_issues(
        self, 
        keywords: List[str], 
        per_kw: int, 
        max_total: int
    ) -> List[RawIssue]:
        """
        Search for GitHub issues using provided keywords.
        
        Args:
            keywords: List of keywords to search for
            per_kw: Maximum issues per keyword
            max_total: Maximum total issues across all keywords
            
        Returns:
            List of RawIssue objects
        """
        loop = asyncio.get_running_loop()
        all_issues = []
        total_collected = 0
        
        for keyword in keywords:
            if total_collected >= max_total:
                break
                
            # Build GitHub search query
            query = f'is:issue is:open in:title,body "{keyword}" sort:updated-desc'
            
            # Calculate how many issues to collect for this keyword
            remaining_total = max_total - total_collected
            issues_to_collect = min(per_kw, remaining_total)
            
            # Execute search asynchronously
            github_issues = await loop.run_in_executor(
                None, 
                self._search_issues_sync, 
                query, 
                issues_to_collect
            )
            
            # Convert to RawIssue objects
            for github_issue in github_issues:
                if total_collected >= max_total:
                    break
                    
                raw_issue = self._convert_to_raw_issue(github_issue)
                all_issues.append(raw_issue)
                total_collected += 1
        
        return all_issues
    
    def _search_issues_sync(self, query: str, limit: int) -> List[Issue]:
        """
        Synchronous GitHub search to be run in executor.
        
        Args:
            query: GitHub search query string
            limit: Maximum number of issues to return
            
        Returns:
            List of GitHub Issue objects
        """
        search_result = self.client.search_issues(query=query)
        issues = []
        
        try:
            for issue in search_result:
                if len(issues) >= limit:
                    break
                issues.append(issue)
        except Exception:
            # Handle potential GitHub API errors gracefully
            pass
            
        return issues
    
    def _convert_to_raw_issue(self, github_issue: Issue) -> RawIssue:
        """
        Convert GitHub Issue object to RawIssue schema.
        
        Args:
            github_issue: GitHub Issue object from PyGithub
            
        Returns:
            RawIssue object
        """
        return RawIssue(
            id=github_issue.number,
            url=github_issue.html_url,
            title=github_issue.title,
            body=github_issue.body or "",  # Handle None body
            labels=[label.name for label in github_issue.labels],
            repo=github_issue.repository.full_name
        ) 