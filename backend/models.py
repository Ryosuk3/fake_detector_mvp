from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

# Модели для внутреннего использования

class Claim(BaseModel):
    text: str
    importance: float = 1.0
    entities: List[str] = []


class Entity(BaseModel):
    text: str
    type: str  # PER, ORG, LOC, DATE, etc.
    start: int
    end: int


class Article(BaseModel):
    url: str
    domain: str
    title: str
    date: Optional[str] = None
    text: str
    chunks: List[str] = []
    trust_level: float = 1.0


class VerificationRequest(BaseModel):
    text: str
    claims: List[Claim] = []
    entities: List[Entity] = []


class VerificationResult(BaseModel):
    status: str
    confidence: float
    claim_results: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    warnings: List[str] = []

