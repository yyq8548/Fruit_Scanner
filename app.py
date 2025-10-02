import streamlit as st
from PIL import Image
from agent.fruit_agent import FruitScannerAgent

st.set_page_config(page_title="FreshSense AI", page_icon="🍎", layout="centered")
st.title("🍎 FreshSense AI")
st.caption("Agentic computer vision assistant for produce quality assessment")

st.write(
    "Upload a fruit image. The agent checks image quality, runs DenseNet201 inference, "
    "evaluates confidence, and returns an actionable freshness recommendation."
)

agent = FruitScannerAgent(model_path="models/densenet201.h5")
uploaded_file = st.file_uploader("Upload fruit image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")

    with st.spinner("FreshSense Agent is analyzing..."):
        state = agent.run(image)

    st.subheader("Agent Decision")
    st.write(f"**Decision:** {state.decision}")
    st.write(f"**Status:** {state.status}")

    st.subheader("Prediction")
    if state.prediction:
        st.write(f"**Class:** {state.prediction.class_name}")
        st.write(f"**Confidence:** {state.prediction.confidence:.2%}")
    else:
        st.write("No prediction was generated because the agent stopped early.")

    st.subheader("Image Quality")
    if state.quality:
        st.write(f"**Brightness:** {state.quality.brightness:.2f}")
        st.write(f"**Edge Strength:** {state.quality.edge_strength:.2f}")
        st.write(f"**Dark:** {state.quality.is_dark}")
        st.write(f"**Blurry:** {state.quality.is_blurry}")
        st.write(f"**Overexposed:** {state.quality.is_overexposed}")

    if state.warnings:
        st.warning("\n".join(state.warnings))

    st.subheader("Recommendation")
    st.success(state.recommendation)

    with st.expander("Agent Trace"):
        for step in state.trace:
            st.write(f"- {step}")
else:
    st.info("Upload a fruit photo to start.")
