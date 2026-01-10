from agent.state import AgentState


class Planner:
    """
    Planner decides the next action for the agent based on the current AgentState.

    This makes the workflow decision-based instead of a hard-coded pipeline.
    """

    def plan_after_quality_check(self, state: AgentState) -> str:
        if state.quality is None:
            return "run_quality_check"

        if state.quality.is_dark or state.quality.is_blurry or state.quality.is_overexposed:
            return "request_retake"

        return "run_vision"

    def plan_after_inference(self, state: AgentState) -> str:
        if state.prediction is None:
            return "request_retake"

        if state.prediction.confidence < 0.70:
            return "request_retake"

        return "generate_recommendation"
