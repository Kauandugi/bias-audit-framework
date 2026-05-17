# 🔍 BiasAuditFW: Auditing Cultural & Demographic Bias in Generative AI

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Computer Vision](https://img.shields.io/badge/Domain-Computer_Vision-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

BiasAuditFW is an automated, open-source auditing framework designed to quantify representational and cultural biases in text-to-image diffusion models (e.g., Stable Diffusion, Midjourney, DALL-E, NanoBanana). 

This project transitions bias auditing from qualitative observation to a rigorous quantitative pipeline, exposing the "Patchwork Effect" and the aesthetic erasure of the Global South in synthetic media.

## 🧠 Architecture and Methodology

The framework operates on a dual-axis auditing pipeline:
1. **Demographic Extraction (DeepFace):** Utilizes the `RetinaFace` backend to extract facial features, transforming unstructured pixel data into categorical variables (Race and Gender) and isolating the algorithmic "default whiteness" and male dominance.
2. **Zero-Shot Semantic Evaluation (CLIP):** Projects generated images and cultural anchor texts (e.g., *"A typical Latin American university campus"*) into a high-dimensional latent space. By calculating cosine similarity, it mathematically proves whether models possess the cultural competence to depict non-Eurocentric realities.

## 📊 Statistical Rigor
The pipeline automatically runs non-parametric hypothesis testing (**Mann-Whitney U Test**, $\alpha = 0.05$) to compare control groups (Neutral Prompts) against treatment groups (Inclusive Prompts). Our findings demonstrate that inclusive prompt engineering is often statistically insufficient to overcome ingrained Eurocentric architectural and cultural baselines.

## 🚀 Repository Structure

* `notebooks/`: Contains the Google Colab environment with the core extraction pipeline.
* `app.py`: An interactive Streamlit Dashboard for data visualization.
* `data/`: The validated `.csv` dataset generated during the extraction phase.
* `docs/`: Academic charts (Seaborn/Matplotlib) ready for publication.

## 💻 How to Run the Interactive Dashboard

You can explore the generated data and statistical reports locally using Streamlit:

```bash
# Clone the repository
git clone [https://github.com/your-username/BiasAuditFW.git](https://github.com/your-username/BiasAuditFW.git)
cd BiasAuditFW

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py

```

This framework is being developed as a Bachelor's Thesis in Computer Science at the State University of Mato Grosso do Sul (UEMS), expanding upon qualitative research conducted during an academic exchange at the Universidad Nacional de Colombia (UNAL).

Author: Kauan Henrick Teixeira da Silva

Role: Software Developer | Python & Computer Vision
