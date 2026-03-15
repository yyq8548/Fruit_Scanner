from PIL import Image

from agent.state import AgentState
from agent.planner import Planner
from tools.image_quality import ImageQualityTool
from tools.vision import DenseNetVisionTool
from tools.confidence import ConfidenceTool
from tools.reasoning import RuleBasedReasoningTool
from tools.recommendation import RecommendationTool


class FruitScannerAgent:
    """
    Tool-orchestrating agent for fruit freshness analysis.

    The planner decides which step should happen next based on AgentState.
    """

    def __init__(self, model_path: str, min_confidence: float = 0.70):
        self.planner = Planner()
        self.quality_tool = ImageQualityTool()
        self.vision_tool = DenseNetVisionTool(model_path=model_path)
        self.confidence_tool = ConfidenceTool(min_confidence=min_confidence)
        self.reasoning_tool = RuleBasedReasoningTool()
        self.recommendation_tool = RecommendationTool()

    def run(self, image: Image.Image) -> AgentState:
        state = AgentState(image=image)
        state.add_trace("Agent initialized.")

        state = self.quality_tool.run(state)

        next_action = self.planner.plan_after_quality_check(state)
        state.add_trace(f"Planner selected next action after quality check: {next_action}.")

        if next_action == "request_retake":
            state.decision = "retake_photo"
            state.status = "stopped_due_to_image_quality"
            state.add_trace("Agent stopped before inference based on planner decision.")
            state = self.reasoning_tool.run(state)
            state = self.recommendation_tool.run(state)
            return state

        state = self.vision_tool.run(state)
        state = self.confidence_tool.run(state)

        next_action = self.planner.plan_after_inference(state)
        state.add_trace(f"Planner selected next action after inference: {next_action}.")

        if next_action == "request_retake":
            state.decision = "retake_photo"
            state.status = "low_confidence"
            state = self.reasoning_tool.run(state)
            state = self.recommendation_tool.run(state)
            return state

        state = self.reasoning_tool.run(state)
        state = self.recommendation_tool.run(state)
        state.add_trace("Agent workflow completed.")
        return state
