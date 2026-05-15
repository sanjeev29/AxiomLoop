from typing import Literal

from pydantic import BaseModel, Field


class ResearchFinding(BaseModel):
    source_url: str
    key_claim: str
    evidence_strength: float = Field(ge=0, le=1)
    contradicts_previous: bool = False


class ResearchState(BaseModel):
    query: str
    findings: list[ResearchFinding] = []
    steps_taken: int = 0
    max_steps: int = 5
    is_complete: bool = False


class Paper(BaseModel):
    title: str
    abstract: str | None = None
    url: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    citations: int | None = None
    venue: str | None = None
    pdf_url: str | None = None
    source: Literal["arxiv", "scholar", "web"]


class WebSearch(BaseModel):
    title: str
    url: str
    snippet: str | None = None
    score: float | None = None


class PageContent(BaseModel):
    url: str
    title: str | None = None
    text: str
    author: str | None = None
    date: str | None = None


class Note(BaseModel):
    """One structured evidence claim. Tuple shape lets us group/contradict."""
    id: int
    source_url: str
    subject: str
    predicate: str
    object: str
    quote: str | None = None
    created_at: str


class ClaimGroup(BaseModel):
    """Notes bucketed by (subject, predicate). status flags contradictions."""
    subject: str
    predicate: str
    status: Literal["single", "agreement", "contradiction"]
    entries: list[Note]


class Position(BaseModel):
    stance: str
    sources: list[str] = Field(default_factory=list)


class Disagreement(BaseModel):
    topic: str
    positions: list[Position]


class VerifierReport(BaseModel):
    agreements: list[str] = Field(default_factory=list)
    disagreements: list[Disagreement] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    remark: str = ""
