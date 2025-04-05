import os
import numpy as np
from PIL import Image

# Class order should match the training dataset folder order.
CLASS_NAMES = [
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges",
]


class FruitFreshnessModel:
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

    def preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.resize((224, 224))
        arr = np.array(image).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)
        return arr

    def predict(self, image: Image.Image) -> dict:
        """
        If no model file is available, this function returns a demo prediction.
        Replace models/densenet201.h5 with your trained model to enable real inference.
        """
        x = self.preprocess(image)

        if self.model is None:
            return {
                "class_name": "freshapples",
                "confidence": 0.91,
                "raw_probabilities": [0.91, 0.02, 0.02, 0.02, 0.01, 0.02],
            }

        probs = self.model.predict(x)[0]
        idx = int(np.argmax(probs))
        return {
            "class_name": CLASS_NAMES[idx],
            "confidence": float(probs[idx]),
            "raw_probabilities": probs.tolist(),
        }
