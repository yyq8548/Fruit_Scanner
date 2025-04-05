from typing import Dict, Any, List
from PIL import Image
from image_quality import check_image_quality
from model_inference import FruitFreshnessModel


class FruitScannerAgent:
    """
    Agentic workflow around a computer vision classifier.

    The model predicts fruit freshness.
    The agent decides whether the photo is usable, whether confidence is high enough,
    and what recommendation should be returned to the user.
    """

    def __init__(self, model_path: str):
        self.model = FruitFreshnessModel(model_path=model_path)
        self.min_confidence = 0.70

    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        warnings: List[str] = []

        quality = check_image_quality(image)
        if quality["is_dark"]:
            warnings.append("The image appears dark. Retake the photo in brighter lighting for better accuracy.")
        if quality["is_blurry"]:
            warnings.append("The image may be blurry. Retake the photo with a steadier camera for better accuracy.")

        prediction = self.model.predict(image)
        predicted_class = prediction["class_name"]
        confidence = prediction["confidence"]

        if confidence < self.min_confidence:
            decision = "Retake photo"
            reasoning = (
                f"The model predicted {predicted_class}, but confidence is only {confidence:.2%}. "
                "The agent decided not to return a final freshness judgment because the prediction may be unreliable."
            )
            recommendation = "Please retake the photo with clear lighting and make sure the fruit fills most of the image."
        else:
            decision = "Freshness classified"
            freshness = "rotten" if "rotten" in predicted_class.lower() else "fresh"

            reasoning = (
                f"The DenseNet201 inference tool classified the image as {predicted_class} "
                f"with {confidence:.2%} confidence. The agent also checked image quality "
                "before returning the final decision."
            )

            if freshness == "rotten":
                recommendation = "This fruit appears rotten or low quality. I would avoid purchasing or eating it."
            else:
                recommendation = "This fruit appears fresh. It is likely acceptable to purchase or eat."

        return {
            "decision": decision,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "warnings": warnings,
            "reasoning": reasoning,
            "recommendation": recommendation,
            "quality": quality,
        }
