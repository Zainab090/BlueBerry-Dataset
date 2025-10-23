# 🍇 Wild Blueberry Yield Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![ML Project](https://img.shields.io/badge/Category-Machine%20Learning-orange.svg)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## 📚 Table of Contents
1. [Overview](#overview)
2. [Objective](#objective)
3. [Dataset Description](#dataset-description)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Model Development](#model-development)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Feature Importance](#feature-importance)
8. [Results](#results)
9. [Libraries Used](#libraries-used)
10. [Future Work](#future-work)
11. [Author](#author)

---

## 🧩 Overview
This project focuses on predicting the **yield of wild blueberries** based on environmental and pollination factors.  
The dataset is generated from the **Wild Blueberry Pollination Simulation Model**, which models the relationship between pollination efficiency, weather, and plant yield.

---

## 🎯 Objective
Predict the continuous target variable **`yield`** using 17 independent features.

**Evaluation Metric:**  
> Root Mean Squared Error (**RMSE**)

---

## 📊 Dataset Description

| Feature | Description |
|----------|-------------|
| `Row#` | Sample index number |
| `clonesize` | Size of the blueberry clone (plant cluster size) |
| `honeybee`, `bumbles`, `andrena`, `osmia` | Bee species activity levels |
| `MaxOfUpperTRange`, `MinOfUpperTRange`, `AverageOfUpperTRange` | Upper temperature range metrics |
| `MaxOfLowerTRange`, `MinOfLowerTRange`, `AverageOfLowerTRange` | Lower temperature range metrics |
| `RainingDays`, `AverageRainingDays` | Rainfall-related metrics |
| `fruitset`, `fruitmass`, `seeds` | Fruit production and growth metrics |
| `yield` | **Target variable — total blueberry yield (kg/ha)** |

**Rows:** 777  
**Columns:** 18  
**Missing Values:** None ✅  

---

## 🔍 Exploratory Data Analysis (EDA)
- No missing values or data type issues found.  
- Strong correlations detected between:
  - Temperature features (`MaxOfUpperTRange`, `AverageOfUpperTRange`, etc.)
  - Rainfall metrics (`RainingDays`, `AverageRainingDays`)  
- Redundant columns were dropped to reduce multicollinearity.

---

## 🤖 Model Development

### 1️⃣ Data Preparation
```python
X = df.drop(['yield', 'Row#', 'seeds', 'fruitmass', 'fruitset'], axis=1)
y = df['yield']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

lin = LinearRegression()
lin.fit(X_train, y_train)
y_pred_lin = lin.predict(X_test)

print("R²:", round(r2_score(y_test, y_pred_lin), 3))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred_lin)), 3))

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 8, 12, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
print(grid_search.best_params_)

{'max_depth': 8, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 300}


🌿 Feature Importance
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

importances = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(8,5))
sns.barplot(x=importances, y=importances.index)
plt.title("Feature Importance – Random Forest")
plt.show()


Top Predictors of Yield:

🐝 honeybee

🐝 andrena

🌧️ RainingDays

🌡️ AverageOfLowerTRange

🌿 clonesize

📈 Results
Model	R²	RMSE
Linear Regression	0.89	124.4
Random Forest (Tuned)	0.90	150.2

✅ The tuned Random Forest achieved a balanced and generalized performance — no overfitting, stable variance.

📦 Libraries Used

pandas, numpy

matplotlib, seaborn

scikit-learn

xgboost (optional)

GridSearchCV

🚀 Future Work

Experiment with XGBoost, LightGBM, or Gradient Boosting

Apply SHAP/LIME for explainability

Deploy the model using Flask, Streamlit, or FastAPI

Collect real-world field data for model validation

👩‍💻 Author

[Zainab]
Data Scientist


📅 Project Year: 2025
