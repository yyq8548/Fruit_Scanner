from PIL import Image

from agent.planner import Planner
from agent.state import AgentState, ImageQualityResult, PredictionResult


def test_planner_requests_retake_for_dark_image():
    state = AgentState(image=Image.new("RGB", (224, 224), color="black"))
    state.quality = ImageQualityResult(
        brightness=20,
        edge_strength=100,
        is_dark=True,
        is_blurry=False,
        is_overexposed=False,
    )

    planner = Planner()
    assert planner.plan_after_quality_check(state) == "request_retake"


def test_planner_runs_vision_for_good_image():
    state = AgentState(image=Image.new("RGB", (224, 224), color="white"))
    state.quality = ImageQualityResult(
        brightness=120,
        edge_strength=200,
        is_dark=False,
        is_blurry=False,
        is_overexposed=False,
    )

    planner = Planner()
    assert planner.plan_after_quality_check(state) == "run_vision"


def test_planner_requests_retake_for_low_confidence():
    state = AgentState(image=Image.new("RGB", (224, 224), color="white"))
    state.prediction = PredictionResult(
        class_name="freshapples",
        confidence=0.45,
        raw_probabilities=[],
    )

    planner = Planner()
    assert planner.plan_after_inference(state) == "request_retake"


def test_planner_generates_recommendation_for_high_confidence():
    state = AgentState(image=Image.new("RGB", (224, 224), color="white"))
    state.prediction = PredictionResult(
        class_name="freshapples",
        confidence=0.95,
        raw_probabilities=[],
    )

    planner = Planner()
    assert planner.plan_after_inference(state) == "generate_recommendation"
