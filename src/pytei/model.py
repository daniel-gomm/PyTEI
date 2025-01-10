from dataclasses import dataclass
from typing import Optional


@dataclass
class PredictionResult:
    label: str
    score: float

@dataclass
class Rank:
    index: int
    score: float
    text: Optional[str] = None
