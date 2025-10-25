# Model Evaluation Dashboard

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

A Streamlit-based tool for **editing evaluation rubrics**, **generating model outputs**, and **running DeepEval tests** on LLM responses.  
Supports **OpenAI**, **Anthropic (Claude)**, and **Google (Gemini)** models for generation, and uses **DeepEval GEval metrics** for scoring.

---

### 🚀 Run locally

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
````

2. **Start the app**

   ```bash
   streamlit run streamlit_app.py
   ```

---

### 🧩 Features

* **Rubric Editor** – Upload, edit, or create rubrics (`.json`) for DeepEval.
* **Model & Prompt Testing** – Upload test cases (`.csv`), generate outputs from supported LLMs, and evaluate using your rubrics.
* **DeepEval Integration** – Automatically scores and visualizes performance metrics.
* **Visual Reports** – View average scores, pass/fail ratios, and export results as CSV.

```
