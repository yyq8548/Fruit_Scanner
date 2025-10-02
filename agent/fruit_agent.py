from PIL import Image
from agent.state import AgentState
from tools.image_quality import ImageQualityTool
from tools.vision import DenseNetVisionTool
from tools.confidence import ConfidenceTool
from tools.recommendation import RecommendationTool


class FruitScannerAgent:
    """
    Tool-orchestrating agent for fruit freshness analysis.

    DenseNet201 is one tool. The agent coordinates image quality assessment,
    model inference, confidence evaluation, and recommendation generation.
    """

    def __init__(self, model_path: str, min_confidence: float = 0.70):
        self.quality_tool = ImageQualityTool()
        self.vision_tool = DenseNetVisionTool(model_path=model_path)
        self.confidence_tool = ConfidenceTool(min_confidence=min_confidence)
        self.recommendation_tool = RecommendationTool()

    def run(self, image: Image.Image) -> AgentState:
        state = AgentState(image=image)
        state.add_trace("Agent initialized.")

        state = self.quality_tool.run(state)

        # Autonomous decision: stop early if photo quality is too poor.
        if state.quality and (state.quality.is_dark or state.quality.is_blurry):
            state.decision = "retake_photo"
            state.status = "stopped_due_to_image_quality"
            state.add_trace("Agent stopped before inference because image quality was poor.")
            state = self.recommendation_tool.run(state)
            return state

        state = self.vision_tool.run(state)
        state = self.confidence_tool.run(state)
        state = self.recommendation_tool.run(state)
        state.add_trace("Agent workflow completed.")
        return state
