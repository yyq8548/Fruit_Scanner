# FreshSense AI Agent Refactor

This refactor converts the original Streamlit + DenseNet demo into a modular tool-orchestrating agent.

## Architecture

```text
User Image
   ↓
FruitScannerAgent
   ↓
AgentState
   ↓
ImageQualityTool
   ↓
DenseNetVisionTool
   ↓
ConfidenceTool
   ↓
RecommendationTool
   ↓
Final Result
```

## Why This Design

The DenseNet201 model is not the agent. It is one tool used by the agent.

The agent controls the workflow, decides whether the image is good enough for inference, evaluates model confidence, and returns either a final prediction or a retake request.

## Interview Framing

I designed the system around a shared AgentState. Each capability, including image quality assessment, DenseNet201 inference, confidence evaluation, and recommendation generation, was implemented as an independent tool. The FruitScannerAgent orchestrates these tools through a workflow and updates the shared state at each step, making the architecture modular and extensible for future LLM integration.
