import pytest
from pydantic import ValidationError

from app.schemas import (
    SkillProfile,
    RawIssue,
    RankedIssue,
    RecommendationResponse,
    Difficulty
)


class TestSkillProfile:
    def test_valid_skill_profile(self):
        """Test creating a valid SkillProfile."""
        profile = SkillProfile(
            skills=["Python", "FastAPI", "Machine Learning"],
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        
        assert profile.skills == ["Python", "FastAPI", "Machine Learning"]
        assert profile.embedding == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_empty_lists_allowed(self):
        """Test that empty lists are allowed."""
        profile = SkillProfile(skills=[], embedding=[])
        assert profile.skills == []
        assert profile.embedding == []


class TestRawIssue:
    def test_valid_raw_issue(self):
        """Test creating a valid RawIssue."""
        issue = RawIssue(
            id=123,
            url="https://github.com/owner/repo/issues/123",
            title="Fix bug in authentication",
            body="There's a bug in the auth flow...",
            labels=["bug", "authentication"],
            repo="owner/repo"
        )
        
        assert issue.id == 123
        assert str(issue.url) == "https://github.com/owner/repo/issues/123"
        assert issue.title == "Fix bug in authentication"
        assert issue.labels == ["bug", "authentication"]
        assert issue.repo == "owner/repo"

    def test_invalid_url(self):
        """Test that invalid URLs raise ValidationError."""
        with pytest.raises(ValidationError):
            RawIssue(
                id=123,
                url="not-a-valid-url",
                title="Test issue",
                body="Test body",
                labels=[],
                repo="test/repo"
            )


class TestRankedIssue:
    def test_valid_ranked_issue(self):
        """Test creating a valid RankedIssue."""
        issue = RankedIssue(
            id=456,
            url="https://github.com/test/repo/issues/456",
            title="Add new feature",
            body="We need this feature...",
            labels=["enhancement"],
            repo="test/repo",
            score=0.85,
            difficulty="Medium",
            summary="A medium difficulty enhancement request",
            repo_summary="A test repository for software development"
        )
        
        assert issue.score == 0.85
        assert issue.difficulty == "Medium"
        assert issue.summary == "A medium difficulty enhancement request"

    def test_invalid_difficulty(self):
        """Test that invalid difficulty values raise ValidationError."""
        with pytest.raises(ValidationError):
            RankedIssue(
                id=456,
                url="https://github.com/test/repo/issues/456",
                title="Test issue",
                body="Test body",
                labels=[],
                repo="test/repo",
                score=0.5,
                difficulty="Invalid",  # This should fail
                summary="Test summary",
                repo_summary="A test repository"
            )


class TestRecommendationResponse:
    def test_valid_recommendation_response(self):
        """Test creating a valid RecommendationResponse."""
        ranked_issue = RankedIssue(
            id=789,
            url="https://github.com/example/repo/issues/789",
            title="Documentation update",
            body="Update the docs",
            labels=["documentation"],
            repo="example/repo",
            score=0.7,
            difficulty="Easy",
            summary="Easy documentation task",
            repo_summary="An example repository for testing"
        )
        
        response = RecommendationResponse(items=[ranked_issue])
        
        assert len(response.items) == 1
        assert response.items[0].id == 789
        assert response.items[0].difficulty == "Easy"

    def test_empty_items_list(self):
        """Test that empty items list is allowed."""
        response = RecommendationResponse(items=[])
        assert response.items == []


class TestDifficulty:
    def test_difficulty_values(self):
        """Test that all difficulty values are valid."""
        valid_difficulties = ["Easy", "Medium", "Hard"]
        
        for difficulty in valid_difficulties:
            issue = RankedIssue(
                id=1,
                url="https://github.com/test/repo/issues/1",
                title="Test",
                body="Test",
                labels=[],
                repo="test/repo",
                score=0.5,
                difficulty=difficulty,
                summary="Test",
                repo_summary="A test repository"
            )
            assert issue.difficulty == difficulty 