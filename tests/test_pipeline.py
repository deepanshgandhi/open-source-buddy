"""Tests for the recommendation pipeline."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from app.pipeline import recommend
from app.schemas import SkillProfile, RawIssue, RankedIssue, RecommendationResponse, Difficulty


class TestRecommendPipeline:
    @pytest.mark.asyncio
    async def test_recommend_full_pipeline(self):
        """Test the complete recommendation pipeline."""
        # Mock data
        mock_profile = SkillProfile(
            skills=["Python", "FastAPI"],
            embedding=[0.1, 0.2, 0.3]
        )
        
        mock_raw_issues = [
            RawIssue(
                id=1,
                url="https://github.com/test/repo/issues/1",
                title="Python bug fix needed",
                body="Fix Python issue",
                labels=["bug", "python"],
                repo="test/repo"
            ),
            RawIssue(
                id=2,
                url="https://github.com/test/repo/issues/2", 
                title="FastAPI feature",
                body="Add FastAPI feature",
                labels=["enhancement"],
                repo="test/repo"
            )
        ]
        
        mock_ranked_issues = [
            RankedIssue(
                id=1,
                url="https://github.com/test/repo/issues/1",
                title="Python bug fix needed",
                body="Fix Python issue",
                labels=["bug", "python"],
                repo="test/repo",
                score=0.8,
                difficulty="Easy",
                summary="Python bug fix",
                repo_summary="A test repository for Python development"
            )
        ]
        
        with patch('app.pipeline.get_settings') as mock_settings, \
             patch('app.pipeline.GitHubService') as mock_gh_service, \
             patch('app.pipeline.profile_run') as mock_profile_run, \
             patch('app.pipeline.search_run') as mock_search_run, \
             patch('app.pipeline.match_run') as mock_match_run:
            
            # Setup mocks
            mock_settings.return_value.GH_TOKEN = "test_token"
            mock_gh_service_instance = Mock()
            mock_gh_service.return_value = mock_gh_service_instance
            
            mock_profile_run.return_value = mock_profile
            mock_search_run.return_value = mock_raw_issues
            mock_match_run.return_value = mock_ranked_issues
            
            # Call the function
            result = await recommend("I am a Python developer with FastAPI experience", top_k=5)
            
            # Verify the result
            assert isinstance(result, RecommendationResponse)
            assert len(result.items) == 1
            assert result.items[0].score == 0.8
            assert result.items[0].title == "Python bug fix needed"
            
            # Verify function calls
            mock_profile_run.assert_called_once_with("I am a Python developer with FastAPI experience")
            mock_search_run.assert_called_once_with(
                keywords=["Python", "FastAPI"],
                limit=50,  # max(50, 5 * 3)
                svc=mock_gh_service_instance
            )
            mock_match_run.assert_called_once_with(
                profile=mock_profile,
                issues=mock_raw_issues,
                svc=mock_gh_service_instance,
                top_k=5,
                user_text="I am a Python developer with FastAPI experience"
            )
    
    @pytest.mark.asyncio
    async def test_recommend_with_large_top_k(self):
        """Test recommendation with large top_k value."""
        mock_profile = SkillProfile(skills=["Python"], embedding=[0.1])
        
        with patch('app.pipeline.get_settings') as mock_settings, \
             patch('app.pipeline.GitHubService'), \
             patch('app.pipeline.profile_run') as mock_profile_run, \
             patch('app.pipeline.search_run') as mock_search_run, \
             patch('app.pipeline.match_run') as mock_match_run:
            
            mock_settings.return_value.GH_TOKEN = "test_token"
            mock_profile_run.return_value = mock_profile
            mock_search_run.return_value = []
            mock_match_run.return_value = []
            
            # Call with top_k=100
            await recommend("Python developer", top_k=100)
            
            # Verify search_run was called with appropriate limit
            mock_search_run.assert_called_once()
            call_args = mock_search_run.call_args
            assert call_args[1]['limit'] == 300  # 100 * 3
    
    @pytest.mark.asyncio
    async def test_recommend_no_skills_extracted(self):
        """Test recommendation when no skills are extracted."""
        mock_profile = SkillProfile(skills=[], embedding=[])
        
        with patch('app.pipeline.get_settings') as mock_settings, \
             patch('app.pipeline.GitHubService'), \
             patch('app.pipeline.profile_run') as mock_profile_run, \
             patch('app.pipeline.search_run') as mock_search_run, \
             patch('app.pipeline.match_run') as mock_match_run:
            
            mock_settings.return_value.GH_TOKEN = "test_token"
            mock_profile_run.return_value = mock_profile
            mock_search_run.return_value = []
            mock_match_run.return_value = []
            
            result = await recommend("I like cooking and reading books")
            
            assert isinstance(result, RecommendationResponse)
            assert len(result.items) == 0
            
            # Verify search was still called with empty skills
            mock_search_run.assert_called_once_with(
                keywords=[],
                limit=50,
                svc=mock_search_run.call_args[1]['svc']
            )
    
    @pytest.mark.asyncio
    async def test_recommend_profile_agent_failure(self):
        """Test recommendation when profile agent fails."""
        with patch('app.pipeline.get_settings') as mock_settings, \
             patch('app.pipeline.GitHubService'), \
             patch('app.pipeline.profile_run') as mock_profile_run:
            
            mock_settings.return_value.GH_TOKEN = "test_token"
            mock_profile_run.side_effect = Exception("Profile extraction failed")
            
            result = await recommend("Python developer")
            
            # Should return empty response on error
            assert isinstance(result, RecommendationResponse)
            assert len(result.items) == 0
    
    @pytest.mark.asyncio
    async def test_recommend_search_agent_failure(self):
        """Test recommendation when search agent fails."""
        mock_profile = SkillProfile(skills=["Python"], embedding=[0.1])
        
        with patch('app.pipeline.get_settings') as mock_settings, \
             patch('app.pipeline.GitHubService'), \
             patch('app.pipeline.profile_run') as mock_profile_run, \
             patch('app.pipeline.search_run') as mock_search_run:
            
            mock_settings.return_value.GH_TOKEN = "test_token"
            mock_profile_run.return_value = mock_profile
            mock_search_run.side_effect = Exception("Search failed")
            
            result = await recommend("Python developer")
            
            # Should return empty response on error
            assert isinstance(result, RecommendationResponse)
            assert len(result.items) == 0
    
    @pytest.mark.asyncio
    async def test_recommend_match_agent_failure(self):
        """Test recommendation when match agent fails."""
        mock_profile = SkillProfile(skills=["Python"], embedding=[0.1])
        mock_raw_issues = [
            RawIssue(
                id=1,
                url="https://github.com/test/repo/issues/1",
                title="Test issue",
                body="Test body",
                labels=[],
                repo="test/repo"
            )
        ]
        
        with patch('app.pipeline.get_settings') as mock_settings, \
             patch('app.pipeline.GitHubService'), \
             patch('app.pipeline.profile_run') as mock_profile_run, \
             patch('app.pipeline.search_run') as mock_search_run, \
             patch('app.pipeline.match_run') as mock_match_run:
            
            mock_settings.return_value.GH_TOKEN = "test_token"
            mock_profile_run.return_value = mock_profile
            mock_search_run.return_value = mock_raw_issues
            mock_match_run.side_effect = Exception("Matching failed")
            
            result = await recommend("Python developer")
            
            # Should return empty response on error
            assert isinstance(result, RecommendationResponse)
            assert len(result.items) == 0
    
    @pytest.mark.asyncio
    async def test_recommend_default_parameters(self):
        """Test recommendation with default parameters."""
        mock_profile = SkillProfile(skills=["JavaScript"], embedding=[0.1])
        
        with patch('app.pipeline.get_settings') as mock_settings, \
             patch('app.pipeline.GitHubService'), \
             patch('app.pipeline.profile_run') as mock_profile_run, \
             patch('app.pipeline.search_run') as mock_search_run, \
             patch('app.pipeline.match_run') as mock_match_run:
            
            mock_settings.return_value.GH_TOKEN = "test_token"
            mock_profile_run.return_value = mock_profile
            mock_search_run.return_value = []
            mock_match_run.return_value = []
            
            # Call without top_k parameter
            result = await recommend("JavaScript developer")
            
            assert isinstance(result, RecommendationResponse)
            
            # Verify default top_k=10 was used
            mock_match_run.assert_called_once()
            call_args = mock_match_run.call_args
            assert call_args[1]['top_k'] == 10 