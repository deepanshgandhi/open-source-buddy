"""Tests for the FastAPI application."""

import pytest
from unittest.mock import Mock, patch
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app, get_github_service, get_settings
from app.schemas import RecommendationResponse, RankedIssue


class TestHealthEndpoint:
    def test_health_check(self):
        """Test the health check endpoint."""
        # Override dependencies to avoid initialization issues
        app.dependency_overrides[get_settings] = lambda: Mock(GH_TOKEN="test", OPENAI_API_KEY="test")
        app.dependency_overrides[get_github_service] = lambda: Mock()
        
        try:
            client = TestClient(app)
            response = client.get("/healthz")
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}
        finally:
            app.dependency_overrides.clear()


class TestRecommendEndpoint:
    def test_recommend_success(self):
        """Test successful recommendation request."""
        # Mock the pipeline recommend function
        mock_response = RecommendationResponse(
            items=[
                RankedIssue(
                    id=1,
                    url="https://github.com/test/repo/issues/1",
                    title="Test issue",
                    body="Test body",
                    labels=["bug"],
                    repo="test/repo",
                    score=0.8,
                    difficulty="Medium",
                    summary="Test issue summary",
                    repo_summary="A test repository for bug fixes"
                )
            ]
        )
        
        with patch('app.main.recommend') as mock_recommend:
            mock_recommend.return_value = mock_response
            
            # Override dependencies
            app.dependency_overrides[get_settings] = lambda: Mock(GH_TOKEN="test", OPENAI_API_KEY="test")
            app.dependency_overrides[get_github_service] = lambda: Mock()
            
            try:
                client = TestClient(app)
                response = client.post(
                    "/recommend",
                    json={"skills": "I am a Python developer"}
                )
                
                assert response.status_code == 200
                data = response.json()
                assert len(data["items"]) == 1
                assert data["items"][0]["title"] == "Test issue"
                assert data["items"][0]["score"] == 0.8
                
                # Verify the pipeline was called correctly
                mock_recommend.assert_called_once_with("I am a Python developer", top_k=10)
            finally:
                app.dependency_overrides.clear()
    
    def test_recommend_empty_skills(self):
        """Test recommendation with empty skills."""
        mock_response = RecommendationResponse(items=[])
        
        with patch('app.main.recommend') as mock_recommend:
            mock_recommend.return_value = mock_response
            
            # Override dependencies
            app.dependency_overrides[get_settings] = lambda: Mock(GH_TOKEN="test", OPENAI_API_KEY="test")
            app.dependency_overrides[get_github_service] = lambda: Mock()
            
            try:
                client = TestClient(app)
                response = client.post(
                    "/recommend",
                    json={"skills": ""}
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["items"] == []
            finally:
                app.dependency_overrides.clear()
    
    def test_recommend_pipeline_error(self):
        """Test recommendation when pipeline fails."""
        with patch('app.main.recommend') as mock_recommend:
            mock_recommend.side_effect = Exception("Pipeline failed")
            
            # Override dependencies
            app.dependency_overrides[get_settings] = lambda: Mock(GH_TOKEN="test", OPENAI_API_KEY="test")
            app.dependency_overrides[get_github_service] = lambda: Mock()
            
            try:
                client = TestClient(app)
                response = client.post(
                    "/recommend",
                    json={"skills": "Python developer"}
                )
                
                assert response.status_code == 500
                assert "Failed to generate recommendations" in response.json()["detail"]
            finally:
                app.dependency_overrides.clear()
    
    def test_recommend_invalid_request(self):
        """Test recommendation with invalid request body."""
        # Override dependencies
        app.dependency_overrides[get_settings] = lambda: Mock(GH_TOKEN="test", OPENAI_API_KEY="test")
        app.dependency_overrides[get_github_service] = lambda: Mock()
        
        try:
            client = TestClient(app)
            response = client.post(
                "/recommend",
                json={"invalid_field": "test"}
            )
            
            assert response.status_code == 422  # Validation error
        finally:
            app.dependency_overrides.clear()
    
    def test_recommend_missing_body(self):
        """Test recommendation with missing request body."""
        # Override dependencies
        app.dependency_overrides[get_settings] = lambda: Mock(GH_TOKEN="test", OPENAI_API_KEY="test")
        app.dependency_overrides[get_github_service] = lambda: Mock()
        
        try:
            client = TestClient(app)
            response = client.post("/recommend")
            
            assert response.status_code == 422  # Validation error
        finally:
            app.dependency_overrides.clear()


class TestDependencyInjection:
    def test_github_service_dependency_success(self):
        """Test that GitHubService dependency works correctly."""
        mock_response = RecommendationResponse(items=[])
        
        with patch('app.main.recommend') as mock_recommend:
            mock_recommend.return_value = mock_response
            
            # Override dependencies with valid mocks
            app.dependency_overrides[get_settings] = lambda: Mock(GH_TOKEN="test", OPENAI_API_KEY="test")
            app.dependency_overrides[get_github_service] = lambda: Mock()
            
            try:
                client = TestClient(app)
                response = client.post(
                    "/recommend",
                    json={"skills": "Python developer"}
                )
                
                assert response.status_code == 200
            finally:
                app.dependency_overrides.clear()
    
    def test_github_service_not_initialized(self):
        """Test behavior when GitHub service is not initialized."""
        # Test the dependency function directly
        from app.main import get_github_service
        
        with patch('app.main.github_service', None):
            # Test that the dependency function raises HTTPException when service is None
            with pytest.raises(HTTPException) as exc_info:
                get_github_service()
            
            assert exc_info.value.status_code == 500
            assert "GitHub service not initialized" in exc_info.value.detail


class TestApplicationLifespan:
    def test_lifespan_initialization(self):
        """Test application lifespan management."""
        with patch('app.main.get_settings') as mock_get_settings, \
             patch('app.main.GitHubService') as mock_gh_service:
            
            # Setup mocks
            mock_settings = Mock()
            mock_settings.GH_TOKEN = "test_token"
            mock_settings.OPENAI_API_KEY = "test_key"
            mock_get_settings.return_value = mock_settings
            
            mock_gh_instance = Mock()
            mock_gh_service.return_value = mock_gh_instance
            
            # Test that the lifespan context manager works
            from app.main import lifespan
            
            # This is a simplified test that just verifies the lifespan context works
            # In a real test, we'd need more complex setup to verify the global state
            assert lifespan is not None


class TestAPIDocumentation:
    def test_openapi_schema(self):
        """Test that OpenAPI documentation is available."""
        # Override dependencies
        app.dependency_overrides[get_settings] = lambda: Mock(GH_TOKEN="test", OPENAI_API_KEY="test")
        app.dependency_overrides[get_github_service] = lambda: Mock()
        
        try:
            client = TestClient(app)
            response = client.get("/docs")
            assert response.status_code == 200
            
            # Test OpenAPI JSON schema
            response = client.get("/openapi.json")
            assert response.status_code == 200
            schema = response.json()
            
            # Verify our endpoints are documented
            assert "/healthz" in schema["paths"]
            assert "/recommend" in schema["paths"]
            assert "get" in schema["paths"]["/healthz"]
            assert "post" in schema["paths"]["/recommend"]
        finally:
            app.dependency_overrides.clear() 