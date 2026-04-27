# Predicting Customer Churn Using Neural Networks

**A Comparative Study of Neural Network and Machine Learning Models for Customer Churn Prediction within Telecommunications**

Aston University | April 2026

---

## Overview

This project investigates the effectiveness of a Multi-Layer Perceptron (MLP) in predicting customer churn on the Cell2Cell telecommunications dataset, comparing it against traditional and advanced machine learning models. Three research questions are addressed:

- **RQ1** — How does a neural network compare to traditional and advanced ML models at both baseline and tuned levels?
- **RQ2** — How do different class imbalance handling techniques affect the ability to detect churning customers?
- **RQ3** — Which customer features most strongly drive churn predictions, as revealed through SHAP analysis?

---

## Dataset

The Cell2Cell telecommunications churn dataset is used throughout this project.

- **Training set**: `data/raw/cell2celltrain.csv` — 51,047 records, 58 features, 29% churn rate
- **Holdout set**: `data/raw/cell2cellholdout.csv` — 20,000 records, no ground-truth labels

Source: [Duke University Fuqua School of Business](https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom)

---

## Project Structure

```
IP/
├── data/
│   ├── raw/                        # Original Cell2Cell CSV files
│   └── processed/                  # Preprocessed features and results
├── models/
│   ├── preprocessing_pipeline.joblib
│   └── feature_selector.joblib
├── notebooks/
│   ├── 01_data_loading_and_inspection.ipynb
│   ├── 02_data_cleaning_and_preprocessing.ipynb
│   ├── 03_class_imbalance_handling.ipynb
│   ├── 04_baseline_models.ipynb
│   ├── 05_Model_Optimisation.ipynb
│   ├── 06_advanced_models.ipynb
│   ├── 07_neural_network.ipynb
│   ├── 08_shap_analysis.ipynb
│   ├── 09_holdout_evaluation.ipynb
│   └── 10_demo.ipynb
└── README.md
```

---

## Notebooks

| Notebook | Description |
|---|---|
| 01 | Data loading, inspection, class distribution, missing value analysis |
| 02 | Preprocessing pipeline: imputation, encoding, feature selection (SelectKBest K=50), scaling |
| 03 | Class imbalance handling: compares SMOTE, random under-sampling, and class weighting on a Logistic Regression baseline |
| 04 | Baseline models at default settings: Logistic Regression, Decision Tree, Random Forest |
| 05 | Hyperparameter tuning of baseline models using GridSearchCV and RandomizedSearchCV |
| 06 | XGBoost evaluation (default and tuned) and full seven-model cross-model comparison |
| 07 | MLP architecture search, hyperparameter tuning, and final comparison |
| 08 | SHAP analysis applied to XGBoost (Tuned) — global feature importance and beeswarm plots |
| 09 | Five-fold cross-validation on full training set and holdout probability predictions |
| 10 | Demo notebook for running predictions on new customer data |

Run notebooks in order (01 → 09). Each notebook loads preprocessed outputs saved by NB02, so NB02 must be run before NB04 onwards.

---

## Key Results

| Model | F1-score | ROC-AUC |
|---|---|---|
| XGBoost (Tuned) ★ | 0.4893 | 0.6688 |
| Random Forest (Tuned) | 0.4874 | 0.6576 |
| XGBoost (Default) | 0.4742 | 0.6507 |
| MLP (Tuned) | 0.4571 | 0.6295 |
| MLP (Default) | 0.4507 | 0.6245 |
| Decision Tree (Tuned) | 0.4437 | 0.6243 |
| Logistic Regression (Tuned) | 0.4425 | 0.6122 |

**Top churn predictors (SHAP):** CurrentEquipmentDays (0.2918), MonthsInService (0.1701), MonthlyMinutes (0.1214)

---

## Dependencies

```
pandas
numpy
scikit-learn
imbalanced-learn
xgboost
shap
matplotlib
seaborn
joblib
jupyter
```

Install with:

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost shap matplotlib seaborn joblib jupyter
```

---

## Reproducibility

All models use `random_state=42` throughout. The 80/20 train/test split is fixed. Under-sampling is applied inside an `imblearn.Pipeline` at each cross-validation fold to prevent data leakage.
