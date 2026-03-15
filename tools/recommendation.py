from agent.state import AgentState


class RecommendationTool:
    """Generates the final user-facing recommendation from the agent state."""

    def run(self, state: AgentState) -> AgentState:
        if state.decision == "retake_photo":
            state.recommendation = (
                "Please retake the photo with brighter lighting, less blur, "
                "and the fruit clearly centered."
            )
            state.add_trace("RecommendationTool generated retake recommendation.")
            return state

        if state.prediction is None:
            state.recommendation = "No reliable prediction was generated."
            state.add_trace("RecommendationTool generated fallback recommendation.")
            return state

        label = state.prediction.class_name.lower()
        confidence = state.prediction.confidence

        if "rotten" in label:
            state.recommendation = (
                f"The fruit appears rotten or low quality with {confidence:.2%} confidence. "
                "Avoid purchasing or eating it."
            )
        else:
            state.recommendation = (
                f"The fruit appears fresh with {confidence:.2%} confidence. "
                "It is likely acceptable to purchase or eat."
            )

        if state.reasoning:
            state.recommendation += (
                f" Shelf-life estimate: {state.reasoning.shelf_life_estimate}. "
                f"Storage advice: {state.reasoning.storage_advice}"
            )

        state.add_trace("RecommendationTool generated final freshness recommendation.")
        return state
