from typing import Literal
from pydantic import BaseModel, HttpUrl


class SkillProfile(BaseModel):
    skills: list[str]
    embedding: list[float]


class RawIssue(BaseModel):
    id: int
    url: HttpUrl
    title: str
    body: str
    labels: list[str]
    repo: str


Difficulty = Literal["Easy", "Medium", "Hard"]


class RankedIssue(BaseModel):
    id: int
    url: HttpUrl
    title: str
    labels: list[str]
    repo: str
    score: float
    difficulty: Difficulty
    summary: str
    repo_summary: str


class RecommendationResponse(BaseModel):
    items: list[RankedIssue] 