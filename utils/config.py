# FreshSense AI configuration
# Centralizes thresholds, model paths, and UI settings.

APP_TITLE = "FreshSense AI"
APP_ICON = "🍎"
APP_LAYOUT = "centered"

MODEL_PATH = "models/densenet201.h5"
IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges",
]

MIN_CONFIDENCE = 0.70

DARK_THRESHOLD = 60.0
OVEREXPOSED_THRESHOLD = 235.0
BLUR_THRESHOLD = 80.0

MIN_FOREGROUND_RATIO = 0.005
SMALL_FRUIT_RATIO = 0.015
EDGE_THRESHOLD_FLOOR = 20.0
