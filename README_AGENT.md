# Fruit Scanner AI Agent

This upgrade turns the original fruit freshness classifier into an agentic computer vision workflow.

## What the Agent Does

The agent accepts a fruit image, checks whether the photo is usable, runs DenseNet201 inference, evaluates confidence, and returns either a freshness result or a retake request.

## Agent Workflow

```text
User Upload / Camera Image
        ↓
Image Quality Tool
        ├── Lighting check
        └── Blur check
        ↓
DenseNet201 Inference Tool
        ↓
Confidence Decision Logic
        ↓
Fresh / Rotten Result or Retake Request
```

## Tools Used by the Agent

- Streamlit upload interface
- PIL image processing
- Image quality checker
- TensorFlow/Keras DenseNet201 model
- Confidence threshold logic
- Result formatter and recommendation generator

## Autonomous Decisions

The agent makes several decisions without manual intervention:

- Whether the uploaded photo is too dark
- Whether the image may be blurry
- Whether model confidence is high enough
- Whether to return a result or ask the user to retake the photo
- Whether the fruit should be considered fresh or rotten

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Put your trained model here:

```text
models/densenet201.h5
```

Run the app:

```bash
streamlit run app.py
```

## Interview Description

I extended the fruit freshness classifier into an AI agent workflow. The agent accepts fruit photos, checks photo quality, preprocesses the image, calls a DenseNet201 inference model, evaluates prediction confidence, and returns an actionable freshness result. If the lighting is poor, the image is blurry, or confidence is too low, the agent autonomously asks the user to retake the photo instead of returning an unreliable prediction.
