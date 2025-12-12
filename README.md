## 💳 credit-risk-model

**Credit Risk Probability Model for Alternative Data: An End-to-End MLOps Implementation**

### Project Overview

This project implements an end-to-end Machine Learning solution for **Bati Bank** to assess customer creditworthiness for a new **Buy-Now-Pay-Later (BNPL)** service offered in partnership with an eCommerce platform.

The core innovation is the use of non-traditional **Alternative Data** (eCommerce transaction history) to engineer a **credit risk proxy variable** using Recency, Frequency, and Monetary (RFM) analysis. The final product is a production-ready model, deployed via a containerized API, that outputs a risk probability score for new applicants.

### 🚀 Key Deliverables

The goal is to deliver a robust, automated, and reproducible system, encompassing the full MLOps lifecycle:

| Component | Technology / Method | Function |
| :--- | :--- | :--- |
| **Data Processing** | `sklearn.Pipeline`, RFM Analysis, K-Means Clustering | Transforms raw transaction data into model features and defines the `is_high_risk` target variable. |
| **Model Training** | Logistic Regression, Gradient Boosting, WoE/IV | Trains and optimizes predictive models for risk probability. |
| **Experiment Tracking** | MLflow | Logs all parameters, metrics, and artifacts for model governance and selection. |
| **Deployment** | FastAPI, Docker, Uvicorn | Containerizes the final model as a low-latency, scalable prediction API. |
| **Automation** | GitHub Actions (CI/CD), `pytest`, `flake8` | Automates testing, linting, and ensures code quality on every push. |



###  Task 1: Credit Scoring Business Understanding

This section summarizes the regulatory and business context that dictates the model design choices for Bati Bank.

| Topic | Summary |
| :--- | :--- |
| **1. Impact of Basel II** | Regulatory push for **interpretable, stable, and auditable** models. |
| **2. Proxy Variable Necessity** | Required to train a supervised model in the absence of a direct financial default label, using behavioral data (RFM). |
| **3. Simple vs. Complex Trade-Offs** | Balancing high **compliance/trust** (Logistic Regression) against high **predictive performance** (Gradient Boosting). |
***

## Detailed Business Context and Compliance Rationale

This document outlines the key business and regulatory considerations that drive the design and development of a credit risk model, particularly within the context of Basel II requirements and data limitations.

### 1. Impact of Basel II on Model Requirements

The **Basel II Capital Accord** mandates banks to hold a specific amount of capital based on their risk profile. This places rigorous demands on the structure and governance of any risk model, emphasizing *accuracy, transparency, and documentation*.

* **Interpretability:** Models must be easily explainable. Regulated credit environments **discourage "black-box"** models where the reasoning for a prediction is hidden.
* **Documentation & Auditability:** Every step—including data transformations, assumptions, and final model decisions—must be **traceable and auditable**.
* **Stability:** Predictions must be **consistent** over time. Unstable risk estimates (PD, LGD, EAD) can cause large swings in required regulatory capital.

> *In Summary: Basel II necessitates building models that are **Interpretable, Reproducible, and Defensible** to auditors and risk committees.*

### 2. Why a Proxy Default Variable Is Needed

A supervised machine learning model requires a defined outcome (or target variable). When a dataset lacks a clear, actual default outcome, a **Proxy Default Variable** must be engineered.

#### Why a Proxy is Necessary:

* A supervised model (like Logistic Regression) **needs a target variable** to learn from.
* **Behavioral data** (such as Recency, Frequency, and Monetary (RFM) patterns) acts as a practical and timely approximation of an obligor's true credit risk.
* Without a defined proxy, **no risk-prediction model can be trained**.

#### ⚠️ Business Risks of Using a Proxy:

| Risk | Description |
| :--- | :--- |
| **Label Bias** | The engineered proxy may not accurately represent the true, regulatory-defined default behavior, leading to a fundamentally flawed model. |
| **Revenue Loss (Type I Error)** | Mislabeling financially healthy customers ("good") as high-risk ("bad") can lead to unnecessarily low approval rates, resulting in lost revenue. |
| **Regulatory Challenges** | Regulators may challenge or reject a model based on non-financial or behavioral labels that are not directly tied to a contractual default definition. |
| **Model Drift** | Behavioral patterns (RFM) can change over time. If the proxy's meaning weakens, the model's predictive power will degrade. |

> *Conclusion: The proxy must be **carefully engineered, thoroughly validated, and rigorously documented** to maintain its credibility and regulatory acceptance.*

### 3. Trade-Offs: Simple vs. Complex Models

Model selection involves a crucial trade-off between compliance and predictive performance.

| Model Type | Strengths | Weaknesses | Regulatory Impact |
| :--- | :--- | :--- | :--- |
| **Simple, Interpretable Models** (e.g., Logistic Regression, WoE) | Easy to explain, stable, and transparent. | Lower ultimate predictive power and limited capture of non-linear relationships. | **Highly favored** due to inherent interpretability and ease of auditing. |
| **Complex Models** (e.g., Random Forest, Gradient Boosting, XGBoost) | High accuracy, superior capture of complex, non-linear patterns. | Harder to explain, higher risk of overfitting, and more difficult to monitor. | Require **strong justification** and sophisticated explainability tools (e.g., SHAP, LIME) to meet audit standards. |

#### ⚖️ Key Trade-Off:

* **Simple Models:** Offer **more trust and compliance** (high interpretability) at the cost of potentially **lower accuracy**.
* **Complex Models:** Offer **higher performance** (high accuracy) but require **extensive explainability and continuous monitoring** to satisfy regulatory demands.
