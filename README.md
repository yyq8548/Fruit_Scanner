# 🍎 FreshSense AI

```{=html}
<p align="center">
```
# AI Agent for Fruit Freshness Assessment

**DenseNet201 • GPT-5 • Retrieval-Augmented Generation • Streamlit**

Production-style AI system combining computer vision, retrieval, and
large language models for intelligent fruit freshness analysis.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--5-green)
![RAG](https://img.shields.io/badge/RAG-Enabled-success)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![CI](https://img.shields.io/badge/GitHub%20Actions-Passing-brightgreen)

```{=html}
</p>
```

------------------------------------------------------------------------

# ✨ Overview

FreshSense AI is an **agentic computer vision system** that determines
whether fruit is fresh or rotten from an uploaded image.

Unlike a traditional CNN demo, FreshSense combines:

-   🧠 DenseNet201 for visual perception
-   🤖 GPT‑5 for natural language reasoning
-   📚 Retrieval-Augmented Generation (RAG)
-   🛠 Modular tool-based AI agent architecture
-   📈 Confidence estimation
-   🧪 Automated testing with GitHub Actions
-   🔄 Rule-based fallback for reliability

The result is an AI pipeline that performs **perception → planning →
retrieval → reasoning → recommendation**.

------------------------------------------------------------------------

# 🏗 System Architecture

``` text
                 Upload Image
                       │
                       ▼
            Image Quality Tool
                       │
                       ▼
           Scene Analysis Tool
          (Advisory Validation)
                       │
                       ▼
        DenseNet201 Vision Tool
                       │
                       ▼
            Confidence Tool
                       │
                       ▼
     Food Knowledge Retriever (RAG)
                       │
                       ▼
          GPT-5 Reasoning Tool
                       │
                       ▼
          Recommendation Tool
```

------------------------------------------------------------------------

# 🚀 Features

  Capability                      Status
  ------------------------------- --------
  DenseNet201 Transfer Learning   ✅
  Image Quality Assessment        ✅
  Scene Analysis                  ✅
  Tool-based AI Agent             ✅
  Planner                         ✅
  Confidence Evaluation           ✅
  GPT-5 Reasoning                 ✅
  Local RAG Knowledge Base        ✅
  Rule-based Fallback             ✅
  Streamlit UI                    ✅
  Unit Tests                      ✅
  GitHub Actions CI               ✅

------------------------------------------------------------------------

# 🧠 AI Agent Workflow

1.  User uploads a fruit image.
2.  Image Quality Tool checks blur, brightness, and exposure.
3.  Scene Analysis provides advisory feedback.
4.  DenseNet201 predicts freshness.
5.  Confidence Tool validates prediction confidence.
6.  Local RAG retrieves food safety and storage knowledge.
7.  GPT‑5 generates grounded reasoning.
8.  Recommendation Tool produces the final recommendation.

------------------------------------------------------------------------

# 📚 Retrieval-Augmented Generation (RAG)

FreshSense retrieves relevant documents before calling GPT‑5.

Current knowledge includes:

-   Fruit storage
-   Shelf-life guidance
-   Food safety
-   Spoilage symptoms
-   Preservation recommendations

Retrieved knowledge is injected into the GPT prompt to ground the
response.

------------------------------------------------------------------------

# 📂 Project Structure

``` text
FreshSense_AI/
├── agent/
├── tools/
├── utils/
├── tests/
├── data/
├── docs/
├── models/
├── app.py
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

# ▶ Running

``` bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
streamlit run app.py
```

------------------------------------------------------------------------

# 🔑 Enable GPT-5

``` bash
export OPENAI_API_KEY=YOUR_API_KEY
streamlit run app.py
```

If no API key is available, the system automatically falls back to the
built-in rule-based reasoning engine.

------------------------------------------------------------------------

# 🧪 Testing

``` bash
pytest
```

Every push automatically runs the test suite using GitHub Actions.

------------------------------------------------------------------------

# 🛠 Technology Stack

-   Python
-   TensorFlow / Keras
-   DenseNet201
-   Streamlit
-   OpenAI GPT‑5
-   Retrieval-Augmented Generation (RAG)
-   GitHub Actions
-   PyTest
-   NumPy
-   Pillow

------------------------------------------------------------------------

# 🗺 Roadmap

## ✅ Phase 1

-   Computer Vision
-   DenseNet201
-   Streamlit Interface

## ✅ Phase 2

-   Tool-based Agent
-   Planner
-   Scene Analysis
-   Confidence Tool

## ✅ Phase 3

-   GPT‑5 Reasoning
-   Rule-based Fallback

## ✅ Phase 4

-   Local RAG
-   Knowledge Retrieval

## 🔜 Future

-   Embedding-based semantic RAG (FAISS/Chroma)
-   Conversation Memory
-   REST API
-   Multi-Agent Collaboration
-   Cloud Deployment

------------------------------------------------------------------------

# 👨‍💻 Author

**Chris Yu**


------------------------------------------------------------------------

⭐ If you found this project interesting, consider giving it a star!
