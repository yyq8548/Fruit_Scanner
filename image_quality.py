import numpy as np
from PIL import Image, ImageFilter


def check_image_quality(image: Image.Image) -> dict:
    """
    Simple image-quality tool used by the agent.
    Checks brightness and edge strength as lightweight proxies for darkness and blur.
    """
    gray = image.convert("L")
    arr = np.array(gray).astype("float32")

    brightness = float(arr.mean())
    is_dark = brightness < 60

    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_arr = np.array(edges).astype("float32")
    edge_strength = float(edge_arr.var())
    is_blurry = edge_strength < 80

    return {
        "brightness": brightness,
        "edge_strength": edge_strength,
        "is_dark": is_dark,
        "is_blurry": is_blurry,
    }
