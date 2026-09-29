# ET-MLAM-advance-machine-learning_CodeSaviours
Advanced Machine Learning: Ensembles & Hyperparameter Tuning

A hands-on, teacher-style Jupyter Notebook that builds on a previous Scikit-Learn Decision Tree project and introduces ensemble learning, boosting, model comparison, and hyperparameter tuning.

The notebook is designed as a practical learning course: concepts are explained inline, code comments describe the purpose of important steps, and each major section includes exercises before a final real-world retail modeling challenge.

📌 Project Overview

The central question of this notebook is:

Why isn't one Decision Tree enough for a real machine learning project, and how can ensemble methods improve model stability and performance?

The notebook introduces and compares:

Decision Trees

Random Forests

Gradient Boosting

XGBoost

Cross-validation

GridSearchCV

RandomizedSearchCV

Feature importance

Accuracy vs. explainability trade-offs

The final challenge applies these techniques to a retail profitability prediction problem using the Superstore dataset.

🎯 Learning Objectives

By completing this notebook, you will learn how to:

Understand the weaknesses of a single Decision Tree.

Understand bagging and how Random Forest reduces model instability.

Understand boosting and how sequential trees learn from previous errors.

Use GradientBoostingClassifier and XGBClassifier.

Understand the relationship between learning_rate and n_estimators.

Compare models using cross-validation.

Interpret cross-validation standard deviation as a measure of consistency.

Visualize feature importance.

Tune model hyperparameters with GridSearchCV.

Tune larger search spaces with RandomizedSearchCV.

Apply ensemble methods to a real-world retail profitability problem.

Think about the trade-off between predictive accuracy and model explainability.

🧠 What the Notebook Teaches

1. Decision Tree Baseline

The notebook begins with a single DecisionTreeClassifier.

This establishes a baseline model, which is important because more complex models should be compared against a simple reference point.

It also demonstrates how changing max_depth can affect model complexity and potential overfitting.

The first practice exercise compares:

max_depth=2

max_depth=None

using 5-fold cross-validation.

🌲 2. Random Forest — Bagging

Random Forest is introduced as an example of bagging.

The notebook explains that a Random Forest:

Builds many Decision Trees.

Uses random samples of training rows.

Uses random subsets of features at splits.

Combines the trees' predictions.

Reduces the instability of relying on a single tree.

The notebook uses:

RandomForestClassifier(
    n_estimators=200,
    max_depth=4,
    random_state=42
)

It compares Random Forest against the single Decision Tree using both test accuracy and 5-fold cross-validation.

Feature Importance

The notebook also visualizes Random Forest feature importance to show which Wine dataset measurements contribute most to the classification model.

Practice Exercise

A smaller forest with:

n_estimators=5

is compared with the 200-tree version to demonstrate how the number of trees affects stability.

🚀 3. Gradient Boosting

The notebook then introduces boosting.

Unlike Random Forest, where trees are trained independently, boosting builds trees sequentially.

Each new tree attempts to correct mistakes made by the previous trees.

The notebook uses:

GradientBoostingClassifier(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)

The model is evaluated using:

Test accuracy

5-fold cross-validation

⚡ 4. XGBoost

The notebook introduces XGBoost, a widely used implementation of gradient boosting.

Example configuration:

xgb.XGBClassifier(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    eval_metric="mlogloss"
)

The notebook compares XGBoost with the other ensemble methods using the same cross-validation framework.

Learning Rate vs. Number of Trees

A dedicated exercise compares:

learning_rate = 0.01
learning_rate = 0.3

while keeping:

n_estimators = 150

constant.

This demonstrates the important boosting trade-off:

Smaller learning rates make smaller corrections and often require more trees.

Larger learning rates make larger corrections but can increase the risk of overfitting.

📊 5. Fair Model Comparison

The notebook brings the main models together:

Decision Tree

Random Forest

Gradient Boosting

XGBoost

All models are compared using 5-fold cross-validation.

The comparison includes:

Mean accuracy

Standard deviation

Error bars

This reinforces an important machine learning principle:

A small difference in average accuracy does not automatically mean one model is meaningfully better. Variation across folds should also be considered.

🎛️ 6. Hyperparameter Tuning

The notebook then moves from manually selected settings to systematic hyperparameter optimization.

GridSearchCV

GridSearchCV is demonstrated using a Random Forest parameter grid containing:

{
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7, None],
    "min_samples_split": [2, 5, 10]
}

This represents:

3 × 4 × 3 = 36 combinations

With 5-fold cross-validation, the search evaluates the combinations across multiple train/validation splits.

The notebook then evaluates the best Random Forest on the held-out test set.

RandomizedSearchCV

The notebook also introduces RandomizedSearchCV.

Instead of evaluating every possible combination, it samples a specified number of combinations.

Example:

n_iter=20

This is useful when the hyperparameter search space becomes too large for exhaustive Grid Search.

🧪 7. XGBoost Hyperparameter Tuning Exercise

The notebook includes a practical XGBoost tuning exercise using:

param_grid_xgb = {
    "learning_rate": [0.01, 0.1, 0.3],
    "max_depth": [3, 5, 7],
    "n_estimators": [100, 200]
}

The exercise uses 5-fold cross-validation and evaluates the tuned model on the test set.

🛒 8. Final Challenge — Retail Profitability Prediction

The final section moves from the Wine dataset to a real-world-style Superstore retail dataset.

Business Problem

The previous Scikit-Learn introductory project used a single Decision Tree to predict whether a retail order would be profitable.

This notebook rebuilds the same problem using more advanced ensemble techniques.

The target is:

Is_Profitable = (Profit > 0).astype(int)

Input Features

The initial model uses:

Sales

Discount

Quantity

Category

Region

Categorical variables are converted into numerical features using one-hot encoding:

pd.get_dummies(...)

🤖 Retail Models Compared

The final challenge evaluates:

Baseline

Single Decision Tree

DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

Random Forest

RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

XGBoost

XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

The models are evaluated on the same held-out test set using classification accuracy.

🔧 Tuned XGBoost

The notebook then uses RandomizedSearchCV to tune XGBoost.

The search includes:

{
    "n_estimators": randint(100, 400),
    "max_depth": randint(3, 8),
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.85, 1.0]
}

The search uses:

n_iter = 15
cv = 3

The tuned model is then evaluated on the held-out retail test set.

🔎 Retail Feature Importance

The final tuned XGBoost model is used to calculate feature importance.

The notebook visualizes the top 10 features driving order profitability.

This helps connect predictive modeling with a practical business question:

Which characteristics of an order appear most useful for predicting whether it will be profitable?

🧩 Final Practice Task

The final exercise extends the retail model in two ways.

1. Add Shipping Mode

Ship Mode is added as another categorical feature and one-hot encoded.

2. Increase Randomized Search

The hyperparameter search is repeated with:

n_iter=30

instead of:

n_iter=15

The purpose is to investigate whether a larger search produces a meaningful improvement.

3. Business Interpretation

The notebook finishes by asking the learner to think beyond accuracy.

The key question is:

Is a more complex model justified if the improvement in accuracy is small but the model becomes harder for non-technical stakeholders to understand?

This introduces the practical machine learning trade-off between:

Predictive performance

Stability

Computational cost

Explainability

Business value

📚 Skills Demonstrated

This notebook demonstrates practical machine learning skills including:

Python

NumPy

Pandas

Matplotlib

Scikit-Learn

XGBoost

Classification

Decision Trees

Ensemble Learning

Bagging

Random Forest

Gradient Boosting

XGBoost

Train/test splitting

Stratified sampling

Cross-validation

Model evaluation

Feature importance

Hyperparameter tuning

Grid Search

Randomized Search

One-hot encoding

Model comparison

Overfitting analysis

Model explainability

🛠️ Technologies & Libraries

The notebook uses:

Python
NumPy
Pandas
Matplotlib
Scikit-Learn
XGBoost
SciPy
Jupyter Notebook

Main Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV
)

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.datasets import load_wine

import xgboost as xgb

📁 Recommended Project Structure

advanced-machine-learning/
│
├── advanced_ml_full_course.ipynb
├── superstore.csv
├── README.md
└── requirements.txt

📦 Installation

Install the required packages with:

pip install numpy pandas matplotlib scikit-learn xgboost scipy jupyter

Or create a requirements.txt file containing:

numpy
pandas
matplotlib
scikit-learn
xgboost
scipy
jupyter

Then run:

pip install -r requirements.txt

▶️ How to Run

1. Start Jupyter Notebook

jupyter notebook

2. Open the notebook

Open:

advanced_ml_full_course.ipynb

3. Run cells in order

The notebook is designed to be executed sequentially.

For the final retail challenge, make sure:

superstore.csv

is available in the notebook's working directory.

🧭 Notebook Roadmap

Section

Topic

0

Why one Decision Tree isn't enough

1

Wine dataset

2

Decision Tree baseline

3

Random Forest / Bagging

4

Gradient Boosting & XGBoost

5

Side-by-side model comparison

6

Hyperparameter tuning

7

Retail profitability modeling

Final

Practice tasks and next steps

🚀 Possible Next Steps

The notebook suggests several natural extensions:

Advanced Boosting

Explore:

LightGBM

CatBoost

Explainability

Explore:

SHAP values

Feature importance

Individual prediction explanations

Deployment

Turn the final model into an interactive application using:

Streamlit

FastAPI

Flask

Further Evaluation

Extend the project beyond accuracy with:

Precision

Recall

F1-score

ROC-AUC

Confusion matrices

Cost-of-error analysis

Cross-validation comparisons

👤 Portfolio Description

You can use the following description for a portfolio or CV:

Advanced Machine Learning — Ensemble Methods & Hyperparameter Tuning: Built and compared Decision Tree, Random Forest, Gradient Boosting, and XGBoost classification models using Python and Scikit-Learn. Applied cross-validation, feature importance analysis, GridSearchCV, and RandomizedSearchCV for model evaluation and tuning. Rebuilt a retail profitability classification problem using ensemble methods and explored the practical trade-off between predictive performance and model explainability.

📌 Key Takeaways

By the end of the notebook, the learner should understand:

Why individual Decision Trees can be unstable.

How Random Forest uses bagging to reduce variance.

How boosting builds models sequentially to correct previous errors.

How learning_rate and n_estimators interact in boosting.

Why cross-validation is preferable to relying on a single train/test split.

Why standard deviation across folds matters.

When Grid Search becomes computationally expensive.

When Randomized Search can be useful.

How to compare baseline and ensemble models fairly.

Why the most accurate model is not automatically the easiest model to explain.

How machine learning techniques can be transferred from a teaching dataset to a business-oriented retail problem.

🎓 Learning Progression

The notebook follows this progression:

Decision Tree
      ↓
Understand instability / overfitting
      ↓
Random Forest
      ↓
Gradient Boosting
      ↓
XGBoost
      ↓
Cross-Validation
      ↓
Model Comparison
      ↓
GridSearchCV
      ↓
RandomizedSearchCV
      ↓
Retail Profitability Model
      ↓
Feature Importance
      ↓
Accuracy vs. Explainability

📄 Project Status

Completed learning project: Advanced ensemble learning and hyperparameter tuning with a final retail profitability modeling challenge.

Primary tools: Python · Pandas · NumPy · Matplotlib · Scikit-Learn · XGBoost · SciPy · Jupyter Notebook
