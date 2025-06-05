import asyncio
import json
from typing import List, Dict
from agents import Agent, Runner
from pydantic import BaseModel
import numpy as np
import os
from openai import AsyncOpenAI

from app.schemas import SkillProfile, RawIssue, RankedIssue, Difficulty
from app.config import get_settings
from app.github_service import GitHubService


class IssueAnalysisOutput(BaseModel):
    difficulty: Difficulty
    summary: str


class RepoSummaryOutput(BaseModel):
    summary: str


class MatchAgent:
    def __init__(self):
        """Initialize the match agent with OpenAI Agents SDK."""
        settings = get_settings()
        self._openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self._repo_summaries: Dict[str, str] = {}  # Cache for repo summaries
        
        # Create the issue analysis agent
        self.analysis_agent = Agent(
            name="Issue Analyzer",
            instructions="""You are a technical issue analysis expert. Analyze the GitHub issue and provide:

1. Difficulty level: "Easy", "Medium", or "Hard"
2. A brief 1-2 sentence summary of what the issue involves

Consider for difficulty:
- Easy: Documentation, simple bug fixes, minor UI changes
- Medium: Feature implementation, moderate bug fixes, testing
- Hard: Complex features, architectural changes, performance issues, security fixes

When User Context is provided, consider the user's stated interests, experience level, and preferences to make the summary more relevant to them. The summary should highlight aspects of the issue that would be most relevant to someone with the user's background and interests.""",
            output_type=IssueAnalysisOutput,
            model="gpt-4o-mini"
        )
        
        # Create the repository summary agent
        self.repo_agent = Agent(
            name="Repository Analyzer",
            instructions="""You are a repository analysis expert. Based on the provided repository information (README content, repository name, and any additional context), create a concise 2-3 sentence summary describing:

1. What the repository/project is about
2. Its main purpose or functionality
3. The primary technology stack or domain

Keep it concise, informative, and accessible to developers who might want to contribute.""",
            output_type=RepoSummaryOutput,
            model="gpt-4o-mini"
        )
    
    async def _get_openai_embedding(self, text: str) -> List[float]:
        """Get embedding using OpenAI API instead of sentence-transformers."""
        try:
            response = await self._openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Failed to get OpenAI embedding: {e}")
            # Return a zero vector as fallback
            return [0.0] * 1536  # text-embedding-3-small has 1536 dimensions
    
    async def _get_chunked_openai_embedding(self, text: str) -> List[float]:
        """
        Get embedding for potentially long text using intelligent chunking.
        
        Args:
            text: Input text (can be very long)
            
        Returns:
            Averaged embedding vector from all chunks
        """
        # Rough estimation: 1 token ≈ 4 characters for safety
        # OpenAI limit: 8192 tokens, so use 6000 chars as safe limit
        MAX_CHUNK_SIZE = 6000
        
        # If text is short enough, use normal embedding
        if len(text) <= MAX_CHUNK_SIZE:
            print(f"📄 Text length {len(text)} chars - using single embedding")
            return await self._get_openai_embedding(text)
        
        print(f"📚 Text length {len(text)} chars - using chunked embedding")
        
        # Split text into chunks with overlap for context preservation
        chunks = []
        overlap = 500  # Character overlap between chunks
        
        for i in range(0, len(text), MAX_CHUNK_SIZE - overlap):
            chunk = text[i:i + MAX_CHUNK_SIZE]
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)
        
        print(f"🔢 Split into {len(chunks)} chunks")
        
        # Get embeddings for each chunk
        chunk_embeddings = []
        for i, chunk in enumerate(chunks):
            try:
                embedding = await self._get_openai_embedding(chunk)
                if embedding and any(x != 0 for x in embedding):  # Valid non-zero embedding
                    chunk_embeddings.append(embedding)
                    print(f"✅ Chunk {i+1}/{len(chunks)} embedded successfully")
                else:
                    print(f"⚠️  Chunk {i+1}/{len(chunks)} returned zero embedding")
            except Exception as e:
                print(f"❌ Failed to embed chunk {i+1}/{len(chunks)}: {e}")
                continue
        
        # Return zero vector if no valid embeddings
        if not chunk_embeddings:
            print("❌ No valid chunk embeddings - returning zero vector")
            return [0.0] * 1536
        
        # Average all chunk embeddings
        chunk_arrays = [np.array(embedding) for embedding in chunk_embeddings]
        averaged_embedding = np.mean(chunk_arrays, axis=0)
        
        print(f"🎯 Averaged {len(chunk_embeddings)} chunk embeddings")
        return averaged_embedding.tolist()
    
    async def run(
        self, 
        profile: SkillProfile, 
        issues: List[RawIssue], 
        svc: GitHubService, 
        top_k: int,
        user_text: str = ""
    ) -> List[RankedIssue]:
        """
        Rank GitHub issues based on skill profile match.
        
        Args:
            profile: User's skill profile with skills and embedding
            issues: List of raw issues to rank
            svc: GitHubService instance (for potential additional data)
            top_k: Number of top ranked issues to return
            user_text: Original user input text for additional context
            
        Returns:
            List of top_k RankedIssue objects sorted by score (highest first)
        """
        if not issues:
            return []
        
        # Process issues in parallel
        ranked_issues = await asyncio.gather(
            *[self._process_issue(profile, issue, svc, user_text) for issue in issues]
        )
        
        # Filter out None results and sort by score
        valid_ranked = [issue for issue in ranked_issues if issue is not None]
        valid_ranked.sort(key=lambda x: x.score, reverse=True)
        
        return valid_ranked[:top_k]
    
    async def _process_issue(self, profile: SkillProfile, issue: RawIssue, svc: GitHubService, user_text: str = "") -> RankedIssue:
        """
        Process a single issue to create a ranked issue.
        
        Args:
            profile: User's skill profile
            issue: Raw issue to process
            svc: GitHubService instance for fetching repo data
            user_text: Original user input text for additional context
            
        Returns:
            RankedIssue with score, difficulty, summary, and repo_summary (without body)
        """
        # Calculate similarity score
        score = await self._calculate_similarity_score(profile, issue)
        
        # Determine difficulty and generate summary using OpenAI Agents SDK
        difficulty, summary = await self._analyze_issue(issue, user_text)
        
        # Get repository summary (cached)
        repo_summary = await self._get_repo_summary(issue.repo, svc)
        
        return RankedIssue(
            id=issue.id,
            url=issue.url,
            title=issue.title,
            labels=issue.labels,
            repo=issue.repo,
            score=score,
            difficulty=difficulty,
            summary=summary,
            repo_summary=repo_summary
        )
    
    async def _calculate_similarity_score(self, profile: SkillProfile, issue: RawIssue) -> float:
        """
        Calculate similarity score between user profile and issue.
        
        Args:
            profile: User's skill profile with embedding
            issue: Raw issue to compare
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not profile.embedding:
            return 0.0
        
        # Create issue text for embedding
        issue_text = f"{issue.title} {issue.body} {' '.join(issue.labels)}"
        
        # Generate embedding for the issue using OpenAI
        issue_embedding = await self._get_chunked_openai_embedding(issue_text)
        
        # Calculate cosine similarity
        profile_vec = np.array(profile.embedding)
        issue_vec = np.array(issue_embedding)
        
        # Handle zero vectors
        profile_norm = np.linalg.norm(profile_vec)
        issue_norm = np.linalg.norm(issue_vec)
        
        if profile_norm == 0 or issue_norm == 0:
            return 0.0
        
        similarity = np.dot(profile_vec, issue_vec) / (profile_norm * issue_norm)
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, float(similarity)))
    
    async def _analyze_issue(self, issue: RawIssue, user_text: str = "") -> tuple[Difficulty, str]:
        """
        Use OpenAI Agents SDK to analyze issue difficulty and generate summary.
        
        Args:
            issue: Raw issue to analyze
            user_text: Original user input text for additional context
            
        Returns:
            Tuple of (difficulty, summary)
        """
        issue_content = f"""Title: {issue.title}

Body: {issue.body[:1000]}...  

Labels: {', '.join(issue.labels)}

Repository: {issue.repo}

User Context: {user_text}"""

        try:
            result = await Runner.run(self.analysis_agent, issue_content)
            analysis_output = result.final_output_as(IssueAnalysisOutput)
            return analysis_output.difficulty, analysis_output.summary
        except Exception:
            # Fallback values
            return "Medium", f"GitHub issue: {issue.title}"
    
    async def _get_repo_summary(self, repo_name: str, svc: GitHubService) -> str:
        """
        Get a summary of the repository by reading README and analyzing the repo.
        Uses caching to avoid repeated API calls for the same repo.
        
        Args:
            repo_name: Repository name in format "owner/repo"
            svc: GitHubService instance for API calls
            
        Returns:
            2-3 sentence summary of the repository
        """
        # Check cache first
        if repo_name in self._repo_summaries:
            return self._repo_summaries[repo_name]
        
        try:
            print(f"📖 Generating summary for repository: {repo_name}")
            
            # Try to get README content
            readme_content = ""
            try:
                # Get repository object
                repo = svc.github.get_repo(repo_name)
                
                # Try different README file names
                readme_files = ["README.md", "README.rst", "README.txt", "README", "readme.md"]
                for readme_file in readme_files:
                    try:
                        readme = repo.get_contents(readme_file)
                        if hasattr(readme, 'decoded_content'):
                            readme_content = readme.decoded_content.decode('utf-8')[:3000]  # Limit to first 3000 chars
                            break
                    except Exception:
                        continue
                
                # If no README found, try to get repository description
                if not readme_content and repo.description:
                    readme_content = f"Repository: {repo.name}\nDescription: {repo.description}"
                    if repo.language:
                        readme_content += f"\nPrimary Language: {repo.language}"
                
            except Exception as e:
                print(f"⚠️  Could not fetch README for {repo_name}: {e}")
                readme_content = f"Repository: {repo_name}"
            
            # Generate summary using AI
            if readme_content.strip():
                repo_info = f"Repository: {repo_name}\n\nREADME Content:\n{readme_content}"
                
                try:
                    result = await Runner.run(self.repo_agent, repo_info)
                    summary_output = result.final_output_as(RepoSummaryOutput)
                    summary = summary_output.summary
                except Exception as e:
                    print(f"⚠️  AI summary failed for {repo_name}: {e}")
                    # Fallback summary
                    if "README" in readme_content and len(readme_content) > 100:
                        summary = f"A software project in the {repo_name} repository. Contains documentation and code for development purposes."
                    else:
                        summary = f"A {repo_name.split('/')[-1]} project repository for software development and collaboration."
            else:
                # Ultimate fallback
                summary = f"A software project repository named {repo_name.split('/')[-1]} available for development and contributions."
            
            # Cache the result
            self._repo_summaries[repo_name] = summary
            print(f"✅ Generated summary for {repo_name}: {summary[:100]}...")
            
            return summary
            
        except Exception as e:
            print(f"❌ Error generating repo summary for {repo_name}: {e}")
            # Fallback summary
            fallback = f"A software project repository named {repo_name.split('/')[-1]} available for development and contributions."
            self._repo_summaries[repo_name] = fallback
            return fallback


# Factory function for easy import
async def run(
    profile: SkillProfile, 
    issues: List[RawIssue], 
    svc: GitHubService, 
    top_k: int,
    user_text: str = ""
) -> List[RankedIssue]:
    """
    Rank GitHub issues based on skill profile match.
    
    Args:
        profile: User's skill profile with skills and embedding
        issues: List of raw issues to rank
        svc: GitHubService instance (for potential additional data)
        top_k: Number of top ranked issues to return
        user_text: Original user input text for additional context
        
    Returns:
        List of top_k RankedIssue objects sorted by score (highest first)
    """
    agent = MatchAgent()
    return await agent.run(profile, issues, svc, top_k, user_text) 