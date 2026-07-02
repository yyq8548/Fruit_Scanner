# 🍎 FreshSense AI

> **An Agentic AI system for fruit freshness assessment using Computer
> Vision, GPT-5, and Retrieval-Augmented Generation (RAG).**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--5-green)
![RAG](https://img.shields.io/badge/RAG-Enabled-success)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![CI](https://img.shields.io/badge/GitHub_Actions-Passing-brightgreen)

------------------------------------------------------------------------

## 🚀 Overview

**FreshSense AI** is a production-style AI agent that evaluates fruit
freshness from an uploaded image.

Instead of relying only on image classification, the system combines:

-   🧠 DenseNet201 computer vision
-   🤖 GPT‑5 reasoning
-   📚 Retrieval-Augmented Generation (RAG)
-   🛠 Modular tool-based agent architecture
-   📈 Confidence estimation
-   🧪 Automated testing with GitHub Actions
-   🔄 Rule-based fallback for robustness

The workflow combines **perception → retrieval → reasoning →
recommendation**.

------------------------------------------------------------------------

## 🏗 Architecture

``` mermaid
flowchart TD
    A[Upload Image]
    B[Image Quality Tool]
    C[Scene Analysis]
    D[DenseNet201 Vision]
    E[Confidence Tool]
    F[Food Knowledge Retriever]
    G[GPT-5 Reasoning]
    H[Recommendation]

    A --> B --> C --> D --> E --> F --> G --> H
```

------------------------------------------------------------------------

## ✨ Features

  Feature            Description
  ------------------ -------------------------------------------------
  🧠 DenseNet201     Fruit freshness classification
  📷 Image Quality   Blur, brightness and exposure detection
  🎯 Confidence      Confidence validation before recommendation
  🤖 GPT‑5           Natural-language explanation
  📚 RAG             Retrieves fruit storage & food safety knowledge
  🛡 Fallback         Rule-based reasoning when LLM is unavailable
  🧪 Testing         PyTest + GitHub Actions CI

------------------------------------------------------------------------

## ⚙️ AI Pipeline

1.  Upload fruit image
2.  Analyze image quality
3.  Perform scene analysis
4.  Predict freshness with DenseNet201
5.  Retrieve relevant food knowledge
6.  Generate grounded reasoning with GPT‑5
7.  Produce shelf-life and storage recommendations

------------------------------------------------------------------------

## 📁 Project Structure

``` text
FreshSense-AI/
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

## ▶️ Quick Start

``` bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
streamlit run app.py
```

### Enable GPT‑5

``` bash
export OPENAI_API_KEY=YOUR_API_KEY
```

------------------------------------------------------------------------

## 🧪 Testing

``` bash
pytest
```

Every push automatically triggers GitHub Actions.

------------------------------------------------------------------------

## 🛠 Tech Stack

-   Python 3.11
-   TensorFlow / Keras
-   DenseNet201
-   OpenAI GPT‑5
-   Streamlit
-   Retrieval-Augmented Generation (RAG)
-   PyTest
-   GitHub Actions

------------------------------------------------------------------------

## 🗺 Roadmap

-   ✅ Computer Vision
-   ✅ Tool-based Agent
-   ✅ GPT‑5 Reasoning
-   ✅ Local RAG
-   ⏳ Embedding-based RAG
-   ⏳ Conversation Memory
-   ⏳ REST API
-   ⏳ Cloud Deployment

------------------------------------------------------------------------

## 👨‍💻 Author

**Yeqiao Yu**

------------------------------------------------------------------------

⭐ If you like this project, please consider giving it a star!
