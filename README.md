# 🍎 FreshSense AI

### Agentic Computer Vision Assistant for Produce Quality Assessment

FreshSense AI is an **agentic computer vision application** that
evaluates fruit freshness from a single image. Rather than exposing a
deep learning model directly, the system is designed around a
**tool-orchestrating AI agent** that autonomously validates image
quality, performs DenseNet201 inference, evaluates prediction
confidence, and generates actionable recommendations.

## Features

-   DenseNet201 fruit freshness classification
-   Tool-based AI Agent architecture
-   Image quality assessment
-   Confidence-aware decision making
-   Streamlit web application
-   Modular architecture for future LLM integration

## Architecture

``` text
User Upload
    ↓
FruitScannerAgent
    ↓
ImageQualityTool
    ↓
DenseNet201 Vision Tool
    ↓
Confidence Tool
    ↓
Recommendation Tool
    ↓
Final Response
```

The DenseNet201 model is one tool---not the agent itself.

## Performance

-   Test Accuracy: **99.7%**
-   Test Loss: **0.014**
-   Transfer Learning with ImageNet
-   Input Resolution: **224×224**

## Tech Stack

-   Python
-   TensorFlow / Keras
-   DenseNet201
-   Streamlit
-   NumPy
-   Pillow

## Roadmap

### Completed

-   DenseNet201 training
-   Streamlit application
-   Tool-based AI Agent
-   Image quality assessment
-   Confidence evaluation

### Next

-   GPT reasoning tool
-   Shelf-life estimation
-   Memory
-   RAG
-   Docker
-   Cloud deployment

## Documentation

See `docs/DEVELOPMENT_LOG.md`

## Resume Summary

Built an agentic computer vision application that orchestrates image
quality assessment, DenseNet201 inference, confidence-based decision
making, and recommendation generation for real-time produce freshness
evaluation.
