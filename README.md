# 📊 Quantifying the Impact of State-Level Opioid Policies on Overdose Mortality in the U.S.
*A Predictive and Policy-Informed Data Science Approach*

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-brightgreen.svg)

---

## 🚀 Overview
This project analyzes the effectiveness of **state-level opioid policies** on overdose mortality across U.S. states (2010–2020).  
By integrating **policy variables (PDMP, Naloxone Access, Medicaid Expansion)** with **socio-economic controls** (poverty, unemployment, median income), we build a **state–year panel dataset**, conduct **policy evaluation (mixed/fixed effects)**, and train **ML models** for prediction and interpretation.  
The goal is to provide **policy-informed, data-driven insights** with an interactive **Streamlit dashboard**.

> **Why it matters:** The U.S. opioid crisis has caused hundreds of thousands of deaths. States adopted varied interventions with mixed outcomes. This project quantifies what works, where, and for whom—through transparent analysis and reproducible code.

---

## 🔗 Live Demo
- **Interactive Dashboard (Streamlit):** 👉 [Click here to view the live app](https://opioid-project-sp39r6jjfkw3nzsw3h8ybt.streamlit.app/)
  
---

## 🔬 Data & Sources
- **Mortality:** CDC WONDER / KFF (opioid overdose death rates)  
- **Policies:** PDAPS, NCSL (PDMP, Naloxone Access, Medicaid Expansion)  
- **Socio-economic:** KFF / Census (poverty, unemployment, median income)  

---

## ✏ Methods

**1) Data Preprocessing**  
- Constructed a **state-year panel dataset (2010–2020)** as the analytical backbone.  
- Encoded major policy variables (e.g., PDMP, Medicaid Expansion, Naloxone Access) as binary indicators (0/1).  
- Merged **socioeconomic covariates** including poverty rate, unemployment, and median household income.  
- Standardized all datasets to a unified `state-year` format (>500 state-year observations).  

**2) Exploratory Data Analysis (EDA)**  
- Generated **yearly mortality trends** and national/state-level heatmaps.  
- Produced **cross-plots** linking poverty, unemployment, and income with overdose mortality.  
- Developed **policy adoption timelines**, highlighting state-level heterogeneity in opioid policy rollout.  
- Created **geo-visualizations** to identify regional mortality clustering (e.g., Appalachia, Midwest).  

**3) Statistical Modeling (Policy Evaluation)**  
- Implemented **Mixed Effects Regression (MixedLM)** with random intercepts at the state level.  
- Controlled for year fixed effects and socioeconomic covariates.  
- Example findings:  
  - *Medicaid Expansion*: coefficient ≈ **+5.55**, *p* < 0.001  
  - *PDMP Implementation*: coefficient ≈ **+3.48**, *p* = 0.003  
  - *Poverty (scaled)*: coefficient ≈ **−2.47**, *p* = 0.012  
- Interpretation: significant coefficients likely reflect **policy lag effects, reporting improvements, or confounding**, not direct causation.  

**4) Machine Learning (Predictive Modeling)**  
- Trained **Random Forest** and **XGBoost** regressors to predict overdose mortality.  
- Best performance: **XGBoost with R² ≈ 0.765** on test data.  
- **Feature importance analysis** identified poverty, PDMP adoption, and Medicaid Expansion as top predictors.  
- Generated **Predicted vs Actual trend plots**, validating model performance and interpretability.
   
---

## 📸 Selected Figures
![Predicted vs Actual 2019](outputs/predicted_vs_actual_2019.png)  
![Predicted vs Actual 2020](outputs/predicted_vs_actual_2020.png)  
![Predicted vs Actual ](outputs/predicted_vs_actual.png)  
![feature importance](outputs/feature_importance.png)  

---

## 🔁 Reproducibility
```bash
  # 1) Create environment
  conda create -n opioid-env python=3.10 -y
  conda activate opioid-env
  
  # 2) Install dependencies
  pip install -r requirements.txt
  
  # 3) (Optional) Rebuild panel data
  python src/merge_data.py
  
  # 4) Run EDA
  python src/eda_analysis.py
  
  # 5) Policy regressions
  python src/mixed_effects_regression.py
  
  # 6) ML forecasting
  python src/ml_forecast.py
  
  # 7) Launch dashboard
  streamlit run src/streamlit_app.py
```
---

## 🛠️ Tech Stack
- **Python**: pandas, numpy, matplotlib, seaborn, statsmodels, scikit-learn, xgboost, streamlit  
- **Visualization**: Streamlit, Plotly  
- **Version control**: GitHub  

---

## 📄 Deliverables
- 📊 Interactive Streamlit dashboard  
- 📘 PDF report  
- 📂 GitHub repository (with full pipeline & outputs)  
- 📝 1-page project summary

---

## ✍️ Author
**Xumo Zhu**  
UC San Diego · B.S. Math–Statistics & Probability (Gaduated in March,2025)  

- **GitHub:** [xumozhu](https://github.com/xumozhu)  
- **LinkedIn:** [Xumo Zhu](https://www.linkedin.com/in/xumo-z-a39b1524b/)  
- **Email:** xumozhu516@gmail.com
