import os
import numpy as np
from PIL import Image

from agent.state import AgentState, PredictionResult
from utils.config import CLASS_NAMES, IMAGE_SIZE


class DenseNetVisionTool:
    """Loads DenseNet201 and performs fruit freshness inference."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

        if os.path.exists(model_path):
            try:
                from tensorflow.keras.models import load_model
                self.model = load_model(model_path)
            except Exception as exc:
                print(f"Could not load model from {model_path}: {exc}")
                print("Running in demo mode.")
        else:
            print(f"Model file not found at {model_path}. Running in demo mode.")

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.resize(IMAGE_SIZE)
        arr = np.array(image).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)
        return arr

    def run(self, state: AgentState) -> AgentState:
        x = self._preprocess(state.image)

        if self.model is None:
            prediction = PredictionResult(
                class_name="freshapples",
                confidence=0.91,
                raw_probabilities=[0.91, 0.02, 0.02, 0.02, 0.01, 0.02],
            )
            state.add_trace("DenseNetVisionTool used demo prediction because no model was loaded.")
        else:
            probs = self.model.predict(x)[0]
            idx = int(np.argmax(probs))
            prediction = PredictionResult(
                class_name=CLASS_NAMES[idx],
                confidence=float(probs[idx]),
                raw_probabilities=probs.tolist(),
            )
            state.add_trace(
                f"DenseNetVisionTool predicted {prediction.class_name} "
                f"with confidence {prediction.confidence:.2%}."
            )

        state.prediction = prediction
        return state
