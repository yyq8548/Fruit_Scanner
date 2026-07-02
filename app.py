import streamlit as st
from PIL import Image

from agent.fruit_agent import FruitScannerAgent
from utils.config import APP_ICON, APP_LAYOUT, APP_TITLE, MODEL_PATH

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout=APP_LAYOUT)
st.title(f"{APP_ICON} {APP_TITLE}")
st.caption("Agentic computer vision assistant for produce quality assessment")

st.write(
    "Upload a fruit image. The agent checks image quality, uses scene analysis as advisory feedback, "
    "runs DenseNet201 inference, reasons over the result, and returns a recommendation."
)

agent = FruitScannerAgent(model_path=MODEL_PATH)
uploaded_file = st.file_uploader("Upload fruit image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")

    with st.spinner("FreshSense Agent is analyzing..."):
        state = agent.run(image)

    st.subheader("Agent Decision")
    st.write(f"**Decision:** {state.decision}")
    st.write(f"**Status:** {state.status}")

    st.subheader("Scene Analysis")
    if state.scene:
        st.write(f"**Foreground Ratio:** {state.scene.foreground_ratio:.4f}")
        st.write(f"**Fruit Too Small:** {state.scene.fruit_is_too_small}")
        st.write(f"**Likely Empty Scene:** {state.scene.likely_empty_scene}")
        st.write(f"**Advisory Only:** Yes — scene analysis does not block inference.")

    st.subheader("Prediction")
    if state.prediction:
        st.write(f"**Class:** {state.prediction.class_name}")
        st.write(f"**Confidence:** {state.prediction.confidence:.2%}")
    else:
        st.write("No prediction was generated because the agent stopped early.")

    st.subheader("Reasoning")
    if state.reasoning:
        st.write(f"**Risk Level:** {state.reasoning.risk_level}")
        st.write(f"**Explanation:** {state.reasoning.explanation}")
        st.write(f"**Shelf-life Estimate:** {state.reasoning.shelf_life_estimate}")
        st.write(f"**Storage Advice:** {state.reasoning.storage_advice}")

    st.subheader("Image Quality")
    if state.quality:
        st.write(f"**Brightness:** {state.quality.brightness:.2f}")
        st.write(f"**Edge Strength:** {state.quality.edge_strength:.2f}")
        st.write(f"**Dark:** {state.quality.is_dark}")
        st.write(f"**Blurry:** {state.quality.is_blurry}")
        st.write(f"**Overexposed:** {state.quality.is_overexposed}")

    if state.structured_warnings:
        for warning in state.structured_warnings:
            if warning.level == "error":
                st.error(warning.message)
            elif warning.level == "suggestion":
                st.info(warning.message)
            else:
                st.warning(warning.message)

    st.subheader("Recommendation")
    st.success(state.recommendation)

    with st.expander("Agent Trace"):
        for step in state.trace:
            st.write(f"- {step}")
else:
    st.info("Upload a fruit photo to start.")
