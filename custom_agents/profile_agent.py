import asyncio
import json
from typing import List
from agents import Agent, Runner
from pydantic import BaseModel
from openai import AsyncOpenAI

from app.schemas import SkillProfile
from app.config import get_settings


class SkillExtractionOutput(BaseModel):
    skills: List[str]


class ProfileAgent:
    def __init__(self):
        """Initialize the profile agent with OpenAI Agents SDK."""
        settings = get_settings()
        self._openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Create the skill extraction agent
        self.skill_agent = Agent(
            name="Skill Extractor",
            instructions="""You are a skill extraction expert. Extract relevant technical skills, programming languages, frameworks, tools, and technologies from the user's text.

Focus on:
- Programming languages (e.g., Python, JavaScript, Go, Rust)
- Frameworks and libraries (e.g., React, FastAPI, Django, TensorFlow, PyTorch)
- Tools and platforms (e.g., Docker, AWS, Git, Kubernetes)
- Technical methodologies (e.g., Machine Learning, DevOps, CI/CD)
- Domain expertise (e.g., Web Development, Data Science, AI/ML)

**IMPORTANT: For general domain terms, expand them to include relevant technologies:**

For AI/ML terms like "agentic ai", "artificial intelligence", "AI agents":
- Include: Python, Machine Learning, TensorFlow, PyTorch, OpenAI, LangChain, Transformers, NLP, Deep Learning

For web development terms like "web apps", "frontend":
- Include: JavaScript, React, Vue, Node.js, HTML, CSS, TypeScript

For backend terms like "APIs", "server":
- Include: Python, FastAPI, Django, Node.js, Express, PostgreSQL, MongoDB

For data terms like "data analysis", "analytics":
- Include: Python, Pandas, NumPy, Jupyter, SQL, Data Science

For DevOps terms like "deployment", "infrastructure":
- Include: Docker, Kubernetes, AWS, CI/CD, Terraform, Jenkins

Be specific and use standard technology names:
- "Python" not "python programming"
- "React" not "react.js"
- "Machine Learning" not "ML"
- "Docker" not "containerization"

Infer relevant technologies from the context.""",
            output_type=SkillExtractionOutput,
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
    
    async def run(self, text: str) -> SkillProfile:
        """
        Extract skills from text and create a skill profile.
        
        Args:
            text: User input text describing their skills, experience, or interests
            
        Returns:
            SkillProfile with extracted skills and their embedding
        """
        # Extract skills using OpenAI Agents SDK
        skills = await self._extract_skills(text)
        
        # Generate embedding for the skills
        embedding = await self._generate_embedding(skills)
        
        return SkillProfile(skills=skills, embedding=embedding)
    
    async def _extract_skills(self, text: str) -> List[str]:
        """
        Use OpenAI Agents SDK to extract relevant technical skills from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of extracted skills
        """
        try:
            result = await Runner.run(self.skill_agent, text)
            skill_output = result.final_output_as(SkillExtractionOutput)
            return skill_output.skills
        except Exception:
            # Fallback to empty list if extraction fails
            return []
    
    async def _generate_embedding(self, skills: List[str]) -> List[float]:
        """
        Generate embedding vector for the skills list.
        
        Args:
            skills: List of skill strings
            
        Returns:
            Embedding vector as list of floats
        """
        if not skills:
            # Return zero vector for empty skills
            return [0.0] * 1536  # text-embedding-3-small dimensions
        
        # Combine skills into a single text for embedding
        skills_text = " ".join(skills)
        
        # Get embedding using OpenAI
        embedding = await self._get_openai_embedding(skills_text)
        
        return embedding


# Factory function for easy import
async def run(text: str) -> SkillProfile:
    """
    Extract skills from text and create a skill profile.
    
    Args:
        text: User input text describing their skills, experience, or interests
        
    Returns:
        SkillProfile with extracted skills and their embedding
    """
    agent = ProfileAgent()
    return await agent.run(text) 