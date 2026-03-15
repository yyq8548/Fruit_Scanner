from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from PIL import Image


@dataclass
class ImageQualityResult:
    brightness: float
    edge_strength: float
    is_dark: bool
    is_blurry: bool
    is_overexposed: bool = False


@dataclass
class PredictionResult:
    class_name: str
    confidence: float
    raw_probabilities: List[float] = field(default_factory=list)


@dataclass
class ReasoningResult:
    explanation: str
    shelf_life_estimate: str
    storage_advice: str
    risk_level: str


@dataclass
class AgentState:
    image: Image.Image
    quality: Optional[ImageQualityResult] = None
    prediction: Optional[PredictionResult] = None
    reasoning: Optional[ReasoningResult] = None
    decision: str = "pending"
    status: str = "initialized"
    recommendation: str = ""
    warnings: List[str] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_trace(self, message: str) -> None:
        self.trace.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)
