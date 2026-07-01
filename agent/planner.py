from agent.state import AgentState
from utils.config import MIN_CONFIDENCE


class Planner:
    """Planner decides the next action from the current AgentState."""

    def plan_after_quality_check(self, state: AgentState) -> str:
        if state.quality is None:
            return "run_quality_check"

        if state.quality.is_dark or state.quality.is_blurry or state.quality.is_overexposed:
            return "request_retake"

        return "run_scene_analysis"

    def plan_after_scene_analysis(self, state: AgentState) -> str:
        if state.scene is None:
            return "run_scene_analysis"

        if state.scene.likely_empty_scene or state.scene.needs_crop_or_closer_photo:
            return "request_retake"

        return "run_vision"

    def plan_after_inference(self, state: AgentState) -> str:
        if state.prediction is None:
            return "request_retake"

        if state.prediction.confidence < MIN_CONFIDENCE:
            return "request_retake"

        return "generate_recommendation"
