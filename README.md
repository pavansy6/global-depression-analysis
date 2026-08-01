# Culturally Aware Depression Risk Prediction Pipeline (Mixture of Experts)

An end-to-end Python machine learning architecture that predicts individual depression risk using a Mixture of Experts (MoE) approach. Instead of forcing a single global formula across incompatible international surveys, this system routes user profiles to specialized regional XGBoost models trained on domain-specific, culturally grounded datasets.

## Why a Multi-Region Mixture of Experts (MoE)?

Many mental health machine learning projects fail because of cultural and systemic differences across populations:
* Applying macro-level country statistics (such as GDP or national weather trends) to predict individual mental health results in the ecological fallacy.
* Symptoms, academic competition, workplace expectations, and self-reported distress carry completely different thresholds across different countries.
* In the United States (CDC NHANES), biological lifestyle markers like sleep disorders, chronic illness, and poverty-to-income ratio dominate predictive splits.
* In India, environmental stressors like academic pressure, work hours, and family expectations are the primary drivers of distress.

### Routing Architecture
```text
      User Profile (Web / App Input)
                    |
                    v
          [ Geographic Router ]
                    |
      +-------------+-------------+
      |             |             |
 Country: IND  Country: USA  Sector: Tech
      |             |             |
      v             v             v
 India Urban   US Clinical   Global Tech
 Model (.pkl)  Model (.pkl)  Model (.pkl)
      |             |             |
      v             v             v
 Local SHAP    PHQ-9 SHAP    Workplace SHAP
 Explanation   Explanation   Explanation

```

## Dataset Specifications

The pipeline ingests three distinct individual-level datasets, standardizing them into clean binary classification targets (1 = At Risk, 0 = Low Risk):

| Dataset | Region | Rows | Target Variable | Key Inputs |
| --- | --- | --- | --- | --- |
| **CDC NHANES** | U.S. (Clinical) | ~5,500 | PHQ-9 Score >= 10 | Sleep Hours, Sleep Disorder, Age, Poverty Ratio |
| **India Survey** | India (Urban) | ~1,000 | Self-Reported Risk | Academic Pressure, Work Hours, Degree, City |
| **OSMI Tech** | Global Tech | ~1,200 | Treatment Sought | Remote Work, Benefits, Work Stress, Family |

## Project Folder Structure

```text
global-depression-analysis/
├── data/
│   ├── raw/                      # Raw .XPT (SAS) and .CSV survey downloads
│   └── processed/                # Cleaned, standardized binary CSVs
├── models/
│   ├── xgboost_us_nhanes.pkl     # Trained US clinical XGBoost model
│   ├── xgboost_india.pkl         # Trained Indian urban XGBoost model
│   ├── xgboost_osmi_tech.pkl     # Trained Global Tech XGBoost model
│   ├── test_data_*.pkl           # Test splits for evaluation
│   └── features_*.pkl            # Feature names for SHAP inference
├── src/
│   ├── download_datasets.py      # Automated dataset downloader & SAS converter
│   ├── 01_data_preprocessing.py  # Data cleaning, encoding & schema alignment
│   ├── 02_model_training.py      # XGBoost training + dynamic class imbalance handling
│   └── 03_evaluate_and_shap.py   # Clinical metrics (ROC-AUC, Recall) & SHAP plots
├── requirements.txt
└── README.md

```

## Installation & Setup

### 1. Requirements

* Python 3.10+
* Package manager: pip or uv

### 2. Install Dependencies

```bash
pip install -r requirements.txt
# OR if using uv:
uv pip install -r requirements.txt

```

### 3. Core Dependencies (requirements.txt)

```text
numpy>=1.26.0
pandas>=2.1.0
scikit-learn>=1.3.0
xgboost>=2.0.0
shap>=0.43.0
imbalanced-learn>=0.11.0
matplotlib>=3.8.0
joblib>=1.3.0
kaggle>=1.6.0

```

## How to Run the Pipeline

Run the scripts sequentially from the root project directory:

### Step 1: Download & Assemble Raw Data

Downloads Kaggle survey files and converts official CDC NHANES .XPT SAS transport files into Pandas-compatible CSVs.

```bash
python src/download_datasets.py

```

Note: If CDC servers block automated connections, manually download DPQ_J.XPT, DEMO_J.XPT, and SLQ_J.XPT from the official CDC NHANES 2017-2018 portal into the data/raw/ directory.

### Step 2: Clean & Standardize Regional Schemas

Handles missing clinical values (CDC codes 7 and 9), encodes string categories, maps binary target variables, and outputs clean CSVs into data/processed/.

```bash
python src/01_data_preprocessing.py

```

### Step 3: Train Regional XGBoost Models

Performs stratified 80/20 train-test splits, computes dynamic class imbalance weights (scale_pos_weight), fits individual XGBoost models, and saves .pkl artifacts into models/.

```bash
python src/02_model_training.py

```

### Step 4: Clinical Evaluation & Explainability (SHAP)

Generates ROC-AUC and Recall reports, and renders interactive SHAP (SHapley Additive exPlanations) summary plots to reveal feature impact per region.

```bash
python src/03_evaluate_and_shap.py

```

## Key Technical Features

* **Avoids Healthcare Black Boxes (XAI):** Uses SHAP TreeExplainer to calculate exact additive feature attribution, ensuring clinicians and end users understand why an individual was flagged as At Risk.
* **Dynamic Imbalance Handling:** Real-world depression prevalence is imbalanced (~10% positive cases). The pipeline automatically calculates and injects scale_pos_weight = count(negative) / count(positive) into each XGBoost classifier to prevent accuracy traps.
* **Directory-Proof Pathing:** Built with dynamic os.path.abspath(**file**) anchoring, allowing terminal execution from any path without relative directory errors.

## Model Generalizability & Clinical Ethics

Disclaimer: This software is an experimental predictive modeling pipeline for educational and research purposes. It is not a diagnostic medical device.

* **Clinical Thresholds:** While the U.S. model utilizes the validated PHQ-9 screening threshold (>= 10), survey-based datasets (India and OSMI Tech) rely on self-reported screening parameters.
* **Deployment Guidance:** Models should never be deployed for clinical decision-making without domain professional oversight, prospective clinical validation, and ethical review regarding patient privacy and subgroup fairness.
'@ | Out-File -FilePath "README.md" -Encoding utf8
