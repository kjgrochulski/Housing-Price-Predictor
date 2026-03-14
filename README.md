Melbourne House Price Prediction
---
Libraries Used

`pandas` · `numpy` · `scikit-learn` · `XGBoost` · `LightGBM` · `matplotlib` · `seaborn` · `missingno` · `scipy`
Predicting residential property prices in Melbourne, Australia by comparing 7 regression models. The best model, **LightGBM**, achieved an **R² of 0.94** on the test set.

---

Dataset

- Source: [Melbourne Housing Snapshot — Kaggle](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot)
- Size: 13,580 property listings across Melbourne suburbs
- Features: 21 columns including location, property type, size, distance from CBD, and sale details

---

Project Structure

```
melbourne-house-price-prediction/
│── melbourne.ipynb       # Main notebook (EDA → Modelling → Results)
│── melb_data.csv         # Dataset 
│── requirements.txt      # Python dependencies
│── README.md
```

---

Key Steps

- **Exploratory Data Analysis** distribution plots, missing value analysis, outlier detection using IQR and z-score
- **Data Cleaning** — handling nulls, removing duplicates, feature extraction from address and date columns
- **Feature Engineering** — created new features including `Building_Density`, `Total_Internal_Rooms`, `Price_Category`, and distance buckets
- **Feature Selection** — used permutation importance and Recursive Feature Elimination (RFE) to identify top predictors
- **Model Training** — compared 7 models using pipelines with `StandardScaler` and `OrdinalEncoder`
- **Hyperparameter Tuning** — optimized each model with `GridSearchCV` using 5-fold cross-validation

---

Results

| Model | R² (Test Set) |
|---|---|
| **LightGBM** | **0.94** |
| XGBoost | ~0.93 |
| Gradient Boosting | ~0.92 |
| Extra Trees | ~0.91 |
| Random Forest | ~0.91 |
| Decision Tree | ~0.85 |
| ElasticNet | ~0.75 |

Tree-based ensemble methods consistently outperformed linear models, likely due to the non-linear relationships between location features and price.

---

How to Run

```bash
# 1. Clone the repo
git clone https://github.com/your-username/melbourne-house-price-prediction.git
cd melbourne-house-price-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset from Kaggle and place melb_data.csv in the root folder

# 4. Open the notebook
jupyter notebook melbourne.ipynb
```
