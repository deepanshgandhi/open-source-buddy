import pytest
from unittest.mock import Mock, patch, AsyncMock
from github.Issue import Issue
from github.Repository import Repository
from github.Label import Label

from app.github_service import GitHubService
from app.schemas import RawIssue


class TestGitHubService:
    def setup_method(self):
        """Set up test fixtures."""
        self.token = "test_token"
        self.service = GitHubService(self.token)
    
    @patch("app.github_service.Github")
    def test_init(self, mock_github):
        """Test GitHubService initialization."""
        service = GitHubService("test_token")
        
        mock_github.assert_called_once_with("test_token", per_page=50)
        assert service.client == mock_github.return_value
    
    def create_mock_issue(
        self, 
        number: int = 123, 
        title: str = "Test Issue",
        body: str = "Test body",
        html_url: str = "https://github.com/test/repo/issues/123",
        labels: list = None,
        repo_name: str = "test/repo"
    ) -> Mock:
        """Create a mock GitHub Issue object."""
        mock_issue = Mock(spec=Issue)
        mock_issue.number = number
        mock_issue.title = title
        mock_issue.body = body
        mock_issue.html_url = html_url
        
        # Mock labels
        if labels is None:
            labels = ["bug", "enhancement"]
        mock_labels = []
        for label_name in labels:
            mock_label = Mock(spec=Label)
            mock_label.name = label_name
            mock_labels.append(mock_label)
        mock_issue.labels = mock_labels
        
        # Mock repository
        mock_repo = Mock(spec=Repository)
        mock_repo.full_name = repo_name
        mock_issue.repository = mock_repo
        
        return mock_issue
    
    def test_convert_to_raw_issue(self):
        """Test conversion from GitHub Issue to RawIssue."""
        mock_issue = self.create_mock_issue(
            number=456,
            title="Fix authentication bug",
            body="There's an issue with login",
            html_url="https://github.com/owner/project/issues/456",
            labels=["bug", "priority-high"],
            repo_name="owner/project"
        )
        
        raw_issue = self.service._convert_to_raw_issue(mock_issue)
        
        assert isinstance(raw_issue, RawIssue)
        assert raw_issue.id == 456
        assert str(raw_issue.url) == "https://github.com/owner/project/issues/456"
        assert raw_issue.title == "Fix authentication bug"
        assert raw_issue.body == "There's an issue with login"
        assert raw_issue.labels == ["bug", "priority-high"]
        assert raw_issue.repo == "owner/project"
    
    def test_convert_to_raw_issue_none_body(self):
        """Test conversion when GitHub issue has None body."""
        mock_issue = self.create_mock_issue(body=None)
        
        raw_issue = self.service._convert_to_raw_issue(mock_issue)
        
        assert raw_issue.body == ""
    
    def test_search_issues_sync(self):
        """Test synchronous search_issues method."""
        # Mock search results
        mock_issues = [
            self.create_mock_issue(number=1),
            self.create_mock_issue(number=2),
            self.create_mock_issue(number=3),
        ]
        
        mock_search_result = Mock()
        mock_search_result.__iter__ = Mock(return_value=iter(mock_issues))
        
        # Mock the client's search_issues method
        self.service.client.search_issues = Mock(return_value=mock_search_result)
        
        results = self.service._search_issues_sync("test query", 2)
        
        assert len(results) == 2
        assert results[0].number == 1
        assert results[1].number == 2
        self.service.client.search_issues.assert_called_once_with(query="test query")
    
    @pytest.mark.asyncio
    async def test_search_issues_single_keyword(self):
        """Test async search_issues with single keyword."""
        mock_issues = [
            self.create_mock_issue(number=1, title="Python bug"),
            self.create_mock_issue(number=2, title="Python feature"),
        ]
        
        # Mock the sync method
        with patch.object(self.service, '_search_issues_sync', return_value=mock_issues) as mock_sync:
            results = await self.service.search_issues(
                keywords=["python"],
                per_kw=5,
                max_total=10
            )
        
        assert len(results) == 2
        assert all(isinstance(issue, RawIssue) for issue in results)
        mock_sync.assert_called_once_with(
            'is:issue is:open in:title,body "python" sort:updated-desc',
            5
        )
    
    @pytest.mark.asyncio
    async def test_search_issues_multiple_keywords(self):
        """Test async search_issues with multiple keywords."""
        # Mock different issues for different keywords
        python_issues = [self.create_mock_issue(number=1, title="Python issue")]
        fastapi_issues = [self.create_mock_issue(number=2, title="FastAPI issue")]
        
        def mock_search_side_effect(query, limit):
            if "python" in query:
                return python_issues
            elif "fastapi" in query:
                return fastapi_issues
            return []
        
        with patch.object(self.service, '_search_issues_sync', side_effect=mock_search_side_effect) as mock_sync:
            results = await self.service.search_issues(
                keywords=["python", "fastapi"],
                per_kw=3,
                max_total=10
            )
        
        assert len(results) == 2
        assert mock_sync.call_count == 2
        
        # Check that queries were built correctly
        calls = mock_sync.call_args_list
        assert 'is:issue is:open in:title,body "python" sort:updated-desc' in calls[0][0]
        assert 'is:issue is:open in:title,body "fastapi" sort:updated-desc' in calls[1][0]
    
    @pytest.mark.asyncio
    async def test_search_issues_max_total_limit(self):
        """Test that max_total limit is respected."""
        # Return more issues than max_total allows
        mock_issues = [
            self.create_mock_issue(number=i) for i in range(1, 6)  # 5 issues
        ]
        
        with patch.object(self.service, '_search_issues_sync', return_value=mock_issues):
            results = await self.service.search_issues(
                keywords=["python"],
                per_kw=10,
                max_total=3  # Should limit to 3 issues
            )
        
        assert len(results) == 3
    
    @pytest.mark.asyncio
    async def test_search_issues_per_kw_limit(self):
        """Test that per_kw limit is respected."""
        # Mock _search_issues_sync to return limited results based on the limit parameter
        def mock_search_limited(query, limit):
            all_mock_issues = [
                self.create_mock_issue(number=i) for i in range(1, 6)  # 5 issues available
            ]
            return all_mock_issues[:limit]  # Return only up to limit
        
        with patch.object(self.service, '_search_issues_sync', side_effect=mock_search_limited):
            results = await self.service.search_issues(
                keywords=["python"],
                per_kw=2,  # Should limit to 2 issues per keyword
                max_total=10
            )
        
        assert len(results) == 2
    
    @pytest.mark.asyncio
    async def test_search_issues_early_termination(self):
        """Test early termination when max_total is reached across keywords."""
        python_issues = [self.create_mock_issue(number=1), self.create_mock_issue(number=2)]
        fastapi_issues = [self.create_mock_issue(number=3), self.create_mock_issue(number=4)]
        
        def mock_search_side_effect(query, limit):
            if "python" in query:
                return python_issues
            elif "fastapi" in query:
                return fastapi_issues
            return []
        
        with patch.object(self.service, '_search_issues_sync', side_effect=mock_search_side_effect) as mock_sync:
            results = await self.service.search_issues(
                keywords=["python", "fastapi"],
                per_kw=5,
                max_total=3  # Should stop after 3 total issues
            )
        
        assert len(results) == 3
        # Should have called both keywords, but stopped collecting after max_total
        assert mock_sync.call_count == 2
    
    @pytest.mark.asyncio
    async def test_search_issues_empty_results(self):
        """Test search_issues with no results."""
        with patch.object(self.service, '_search_issues_sync', return_value=[]):
            results = await self.service.search_issues(
                keywords=["nonexistent"],
                per_kw=5,
                max_total=10
            )
        
        assert results == [] 