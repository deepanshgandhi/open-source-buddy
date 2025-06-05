import pytest
from unittest.mock import Mock, patch, AsyncMock
from our_agents.search_agent import SearchAgent, run as search_run
from app.schemas import RawIssue
from app.github_service import GitHubService


class TestSearchAgent:
    @pytest.mark.asyncio
    async def test_search_agent_run(self):
        """Test SearchAgent run method."""
        # Mock GitHubService
        mock_service = Mock(spec=GitHubService)
        mock_issues = [
            Mock(spec=RawIssue),
            Mock(spec=RawIssue),
        ]
        mock_service.search_issues = AsyncMock(return_value=mock_issues)
        
        agent = SearchAgent()
        results = await agent.run(
            keywords=["python", "fastapi"],
            limit=10,
            svc=mock_service
        )
        
        assert results == mock_issues
        mock_service.search_issues.assert_called_once_with(
            keywords=["python", "fastapi"],
            per_kw=5,  # 10 // 2 keywords
            max_total=10
        )
    
    @pytest.mark.asyncio
    async def test_search_agent_empty_keywords(self):
        """Test SearchAgent with empty keywords."""
        mock_service = Mock(spec=GitHubService)
        
        agent = SearchAgent()
        results = await agent.run(
            keywords=[],
            limit=10,
            svc=mock_service
        )
        
        assert results == []
        mock_service.search_issues.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_search_agent_factory_function(self):
        """Test the factory function."""
        mock_service = Mock(spec=GitHubService)
        mock_issues = [Mock(spec=RawIssue)]
        mock_service.search_issues = AsyncMock(return_value=mock_issues)
        
        results = await search_run(
            keywords=["test"],
            limit=5,
            svc=mock_service
        )
        
        assert results == mock_issues


class TestProfileAgent:
    @pytest.mark.asyncio
    async def test_profile_agent_imports(self):
        """Test that ProfileAgent can be imported and instantiated."""
        from our_agents.profile_agent import ProfileAgent
        
        # Mock the settings to avoid requiring actual API keys
        with patch('our_agents.profile_agent.get_settings') as mock_settings:
            mock_settings.return_value.OPENAI_API_KEY = "test_key"
            mock_settings.return_value.OPENAI_MODEL = "gpt-4o-mini"
            mock_settings.return_value.EMBED_MODEL = "all-MiniLM-L6-v2"
            
            agent = ProfileAgent()
            assert agent is not None


class TestMatchAgent:
    @pytest.mark.asyncio
    async def test_match_agent_imports(self):
        """Test that MatchAgent can be imported and instantiated."""
        from our_agents.match_agent import MatchAgent
        
        # Mock the settings to avoid requiring actual API keys
        with patch('our_agents.match_agent.get_settings') as mock_settings:
            mock_settings.return_value.OPENAI_API_KEY = "test_key"
            mock_settings.return_value.OPENAI_MODEL = "gpt-4o-mini"
            mock_settings.return_value.EMBED_MODEL = "all-MiniLM-L6-v2"
            
            agent = MatchAgent()
            assert agent is not None


class TestAgentsModule:
    def test_agents_imports(self):
        """Test that all agents can be imported from our_agents module."""
        from our_agents import (
            ProfileAgent, 
            SearchAgent, 
            MatchAgent,
            profile_run,
            search_run,
            match_run
        )
        
        assert ProfileAgent is not None
        assert SearchAgent is not None  
        assert MatchAgent is not None
        assert callable(profile_run)
        assert callable(search_run)
        assert callable(match_run) 