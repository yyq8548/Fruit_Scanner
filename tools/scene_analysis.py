import numpy as np
from PIL import ImageFilter

from agent.state import AgentState, SceneAnalysisResult


class SceneAnalysisTool:
    """
    Lightweight scene validation tool.

    This is not a full object detector. It estimates whether the image has enough
    foreground detail for meaningful inference. Later this can be replaced by YOLO
    or another object detection model.
    """

    def __init__(self, min_foreground_ratio: float = 0.005, small_fruit_ratio: float = 0.015):
        self.min_foreground_ratio = min_foreground_ratio
        self.small_fruit_ratio = small_fruit_ratio

    def run(self, state: AgentState) -> AgentState:
        image = state.image.convert("RGB")
        width, height = image.size

        gray = image.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_arr = np.array(edges).astype("float32")

        # Estimate foreground/detail area using edge pixels.
        threshold = max(20.0, float(edge_arr.mean() + edge_arr.std()))
        foreground_mask = edge_arr > threshold
        foreground_ratio = float(foreground_mask.mean())

        likely_empty_scene = foreground_ratio < self.min_foreground_ratio
        fruit_is_too_small = foreground_ratio < self.small_fruit_ratio
        needs_crop_or_closer_photo = likely_empty_scene or fruit_is_too_small

        state.scene = SceneAnalysisResult(
            image_width=width,
            image_height=height,
            foreground_ratio=foreground_ratio,
            fruit_is_too_small=fruit_is_too_small,
            likely_empty_scene=likely_empty_scene,
            needs_crop_or_closer_photo=needs_crop_or_closer_photo,
        )

        state.add_trace(
            f"SceneAnalysisTool completed: foreground_ratio={foreground_ratio:.4f}."
        )

        if likely_empty_scene:
            state.add_warning("The image may not contain enough visible fruit detail.")
        elif fruit_is_too_small:
            state.add_warning("The fruit may be too small in the frame. Move closer or crop the image.")

        return state
