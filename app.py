import streamlit as st
from PIL import Image
from agent import FruitScannerAgent

st.set_page_config(page_title="Fruit Scanner AI Agent", page_icon="🍎", layout="centered")

st.title("🍎 Fresh or Rotten Fruit Scanner AI Agent")
st.write(
    "Upload a fruit photo. The agent checks image quality, runs model inference, "
    "and returns a freshness prediction with an actionable recommendation."
)

agent = FruitScannerAgent(model_path="models/densenet201.h5")

uploaded_file = st.file_uploader("Upload fruit image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")

    with st.spinner("Agent is analyzing the image..."):
        result = agent.analyze(image)

    st.subheader("Result")
    st.write(f"**Decision:** {result['decision']}")
    st.write(f"**Predicted class:** {result['predicted_class']}")
    st.write(f"**Confidence:** {result['confidence']:.2%}")

    st.subheader("Agent Reasoning")
    st.write(result["reasoning"])

    if result["warnings"]:
        st.warning("\n".join(result["warnings"]))

    st.subheader("Recommendation")
    st.success(result["recommendation"])
else:
    st.info("Upload a fruit photo to start.")
