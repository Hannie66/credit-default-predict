# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import time

# Load data and split into features and target
df = pd.read_csv('clean.csv')
features = ['LIMIT_BAL', 'EDUCATION', 'MARRIAGE', 'PAY_1', 'PAY_2', 'PAY_3', 
            'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 
            'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 
            'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'SE_MA', 'AgeBin', 
            'SE_AG', 'Avg_exp_5', 'Avg_exp_4', 'Avg_exp_3', 'Avg_exp_2', 
            'Avg_exp_1', 'Closeness_5', 'Closeness_4', 'Closeness_3', 
            'Closeness_2', 'Closeness_1']
X = df[features].copy()
y = df['def_pay'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
XGB_model = XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1, colsample_bytree=0.8,
    subsample=0.8, min_child_weight=1, random_state=42, n_jobs=-1
)
start = time.time()
XGB_model.fit(X_train, y_train, eval_metric='logloss')
XGB_time = time.time() - start
XGB_y_pred = XGB_model.predict(X_test)

# Train LightGBM model
LGB_model = LGBMClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1, num_leaves=31,
    min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, 
    bagging_freq=5, random_state=42, n_jobs=-1
)
start = time.time()
LGB_model.fit(X_train, y_train)
LGB_time = time.time() - start
LGB_y_pred = LGB_model.predict(X_test)

# Evaluate performance
xgb_metrics = {
    'Accuracy': accuracy_score(y_test, XGB_y_pred),
    'F1': f1_score(y_test, XGB_y_pred),
    'AUC': roc_auc_score(y_test, XGB_model.predict_proba(X_test)[:, 1]),
    'Time': XGB_time
}
lgb_metrics = {
    'Accuracy': accuracy_score(y_test, LGB_y_pred),
    'F1': f1_score(y_test, LGB_y_pred),
    'AUC': roc_auc_score(y_test, LGB_model.predict_proba(X_test)[:, 1]),
    'Time': LGB_time
}

# Print results
print(f"XGBoost Metrics: {xgb_metrics}")
print(f"LightGBM Metrics: {lgb_metrics}")

from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning for XGBoost
XGB_raw_model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42, n_jobs=-1)

# Define parameter grid and perform grid search
xgb_param_grid = {
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 300, 500]
}
xgb_grid = GridSearchCV(estimator=XGB_raw_model, param_grid=xgb_param_grid, scoring='accuracy', cv=5, verbose=2, n_jobs=-1)
xgb_grid.fit(X_train, y_train)

# Best parameters and score
print(f"XGBoost Best Parameters: {xgb_grid.best_params_}")
print(f"XGBoost Best Score: {xgb_grid.best_score_}")

# Hyperparameter tuning for LightGBM
LGB_raw_model = LGBMClassifier(random_state=42, n_jobs=-1)

# Define parameter grid and perform grid search
lgb_param_grid = {
    'num_leaves': [15, 31, 63],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 300, 500]
}
lgb_grid = GridSearchCV(estimator=LGB_raw_model, param_grid=lgb_param_grid, scoring='accuracy', cv=5, verbose=2, n_jobs=-1)
lgb_grid.fit(X_train, y_train)

# Best parameters and score
print(f"LightGBM Best Parameters: {lgb_grid.best_params_}")
print(f"LightGBM Best Score: {lgb_grid.best_score_}")

# Visualizing feature importance
import matplotlib.pyplot as plt

# Get feature importance from both models
xgb_importance = xgb_grid.best_estimator_.feature_importances_
lgb_importance = lgb_grid.best_estimator_.feature_importances_

# Create DataFrames for easy visualization
xgb_df = pd.DataFrame({'Feature': features, 'Importance': xgb_importance}).sort_values(by='Importance', ascending=False)
lgb_df = pd.DataFrame({'Feature': features, 'Importance': lgb_importance}).sort_values(by='Importance', ascending=False)

# Plot feature importance
fig, ax = plt.subplots(1, 2, figsize=(15, 8))
xgb_df.head(10).plot(kind='barh', x='Feature', y='Importance', ax=ax[0], color='darkblue', legend=False)
ax[0].set_title('XGBoost Feature Importance')
lgb_df.head(10).plot(kind='barh', x='Feature', y='Importance', ax=ax[1], color='skyblue', legend=False)
ax[1].set_title('LightGBM Feature Importance')
plt.tight_layout()
plt.show()

# Plot ROC curve
from sklearn.metrics import roc_curve, auc

# Compute ROC curves
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_grid.best_estimator_.predict_proba(X_test)[:, 1])
lgb_fpr, lgb_tpr, _ = roc_curve(y_test, lgb_grid.best_estimator_.predict_proba(X_test)[:, 1])

# Plot ROC curves
plt.figure(figsize=(8, 6))
plt.plot(xgb_fpr, xgb_tpr, label=f'XGBoost (AUC = {auc(xgb_fpr, xgb_tpr):.4f})', color='darkblue')
plt.plot(lgb_fpr, lgb_tpr, label=f'LightGBM (AUC = {auc(lgb_fpr, lgb_tpr):.4f})', color='skyblue')
plt.plot([0, 1], [0, 1], 'k--', lw=0.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.show()

# Compare model performance metrics and training time
models = ['XGBoost', 'LightGBM']
metrics = {
    'Accuracy': [xgb_metrics['Accuracy'], lgb_metrics['Accuracy']],
    'F1 Score': [xgb_metrics['F1'], lgb_metrics['F1']],
    'AUC': [xgb_metrics['AUC'], lgb_metrics['AUC']],
    'Training Time (s)': [xgb_metrics['Time'], lgb_metrics['Time']]
}

# Plotting performance metrics
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# Metrics (Accuracy, F1, AUC)
x = np.arange(len(models))
width = 0.2
ax[0].bar(x - width, metrics['Accuracy'], width=width, label='Accuracy', color='orange')
ax[0].bar(x, metrics['F1 Score'], width=width, label='F1 Score', color='salmon')
ax[0].bar(x + width, metrics['AUC'], width=width, label='AUC', color='skyblue')
ax[0].set_xticks(x)
ax[0].set_xticklabels(models)
ax[0].set_title('Performance Metrics')
ax[0].legend()

# Training time
ax[1].bar(models, metrics['Training Time (s)'], color='darkblue')
ax[1].set_title('Training Time (s)')

plt.tight_layout()
plt.show()

# Discussion of feature importance
print("Top 10 Features by Importance - XGBoost")
print(xgb_df.head(10))

print("Top 10 Features by Importance - LightGBM")
print(lgb_df.head(10))