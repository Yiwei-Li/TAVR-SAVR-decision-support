# Aortic Valve Treatment Decision Support Tool


**Live App**: https://tavr-savr-decision-support.streamlit.app/

This is an interactive clinical decision support tool designed to help visualize and compare the risk of major adverse cardiovascular events (MACE) for patients undergoing either **Transcatheter Aortic Valve Replacement (TAVR)** or **Surgical Aortic Valve Replacement (SAVR)**.

<br>

## 🔍 Purpose of This Project 

Making informed decisions between **Transcatheter Aortic Valve Replacement (TAVR)** and **Surgical Aortic Valve Replacement (SAVR)** is complex and highly individualized. While national guidelines and static decision aids exist, they often provide population-level guidance that lacks personalization and transparency.


This project was built to address the gap by:

- Simulate an individual patient profile
- Visualize that patient’s similarity to past cases
- Compare predicted outcomes under both treatments
- Understand which baseline characteristics drive risk
- Receive a model-guided treatment recommendation (not a decision)

<br>

## 📊 Data Description

- **Source**: Real-world, de-identified clinical data collected from the **Northern New England region**
- **Timeframe**: 2015–2023
- **Patients**: >5,000 individuals treated for aortic valve disease
- **Inclusion**:
  - First-time valve procedures only
  - Elective and urgent cases
  - No missing data (complete-case analysis only)

The dataset includes over 20 baseline clinical features and outcomes used to train predictive models and estimate individualized treatment effects.

<br>

## ⚠️ Disclaimers

> ❗ This tool is **not** a diagnostic device, nor is it intended to replace clinical judgment.

- It is for **educational, exploratory, and decision-support purposes only**
- Risk estimates are based on patterns observed in historical data from a specific geographic region
- These predictions **may not generalize** to other populations or clinical settings
- Always consult a healthcare professional before making medical decisions

<br>

## 🔒 Limitations

- **Data Quality**: This model was trained on a subset (~30%) of the original dataset due to missing values and it's unclear whether the missingness reflects true absence or inconsistencies in data entry, potentially introducing bias.
- **Geographic Generalizability**: The dataset reflects patients treated in the **Northern New England region**. Practices, patient populations, and outcomes may differ in other areas, limiting broader applicability.
- **Model Complexity**: The current version uses default XGBoost settings for both outcome and treatment effect models. While robust, this model is not fine-tuned and may benefit from hyperparameter optimization, cross-validation, or utilization of alternative models.

<br>

## 💭 Future Work

- **Leverage Full EHR Integration**: While this tool uses structured EHR data, much of the richness of patient information, including physician notes, imaging results, and previous labs, exists in unstructured formats. A key future step is to extract complete and context-rich data to improve the real-time accuracy of predictions.
- **Quantify Uncertainty**: Display confidence intervals to treatment effect estimates when the new patient is far from the training data manifold.
- **Expand Generalizability**: Explore **federated learning** to securely aggregate models across institutions from other regions.

<br>

## 📬 Contact

For questions, feedback, or collaboration, feel free to reach out via GitHub Issues or Yiwei.Li@hitchcock.org