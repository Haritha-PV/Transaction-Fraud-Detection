# Import necessary libraries
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib

# Load data from SQLite database
conn = sqlite3.connect('Database.db')
df = pd.read_sql_query('SELECT * FROM Fraud_detection', conn)
conn.close()

# Display basic info
print("Initial DataFrame Info:")
df.info()

# Data Preprocessing
print("\n### Data Preprocessing ###")
# Convert categorical columns to numeric using one-hot encoding
df = pd.get_dummies(df, columns=['type'], drop_first=True)

# Drop unnecessary columns
df.drop(columns=['nameOrig', 'nameDest'], inplace=True)

# Convert numeric columns to proper numeric types
numeric_columns = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                   'oldbalanceDest', 'newbalanceDest', 'isFraud', 'isFlaggedFraud']

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle missing values
df.fillna(0, inplace=True)

# Convert boolean columns to integers for compatibility
boolean_columns = ['type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
df[boolean_columns] = df[boolean_columns].astype(int)

# Verify preprocessing
print("\nDataFrame Info After Preprocessing:")
df.info()

# Exploratory Data Analysis
print("\n### Exploratory Data Analysis ###")
plt.figure(figsize=(6, 6))
df['isFraud'].value_counts().plot(kind='pie', autopct='%1.2f%%', labels=['Non-Fraud', 'Fraud'], colors=['lightblue', 'red'])
plt.title('Fraudulent vs Non-Fraudulent Transactions')
plt.show()

plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlations')
plt.show()

# Feature Engineering
print("\n### Feature Engineering ###")
df['balanceDifference'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['destinationBalanceDiff'] = df['oldbalanceDest'] - df['newbalanceDest']
df['net_transaction'] = df['amount'] - (df['balanceDifference'] - df['destinationBalanceDiff'])
df['origin_amount_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
df['dest_amount_ratio'] = df['amount'] / (df['oldbalanceDest'] + 1)

# Drop features with high correlation
features_to_drop = ['newbalanceOrig', 'newbalanceDest', 'origin_amount_ratio', 'step']
df.drop(features_to_drop, axis=1, inplace=True)

# Dataset Splitting
print("\n### Dataset Splitting ###")
train_data = df.iloc[:4000000]
eval_data = df.iloc[4000000:5000000]
prod_data = df.iloc[5000000:]

X_train = train_data.drop('isFraud', axis=1)
y_train = train_data['isFraud']
X_eval = eval_data.drop('isFraud', axis=1)
y_eval = eval_data['isFraud']
X_test = prod_data.drop('isFraud', axis=1)
y_test = prod_data['isFraud']

# Handle Class Imbalance using SMOTE
print("\n### Handling Class Imbalance with SMOTE ###")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Model Training and Hyperparameter Tuning
print("\n### Model Training and Tuning ###")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [6, 9],
    'learning_rate': [0.1],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}
xgb_model = XGBClassifier(random_state=42, scale_pos_weight=len(y_train) / (2 * sum(y_train)), eval_metric='logloss')

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=2,
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train_resampled, y_train_resampled)

# Evaluate the best model
best_model = grid_search.best_estimator_

# Training Metrics
y_train_pred = best_model.predict(X_train_resampled)
y_train_prob = best_model.predict_proba(X_train_resampled)[:, 1]
train_classification_report = classification_report(y_train_resampled, y_train_pred)
train_roc_auc = roc_auc_score(y_train_resampled, y_train_prob)
print("\nTraining Classification Report:\n", train_classification_report)
print(f"Training ROC-AUC Score: {train_roc_auc:.4f}")

# Evaluation Metrics
y_eval_pred = best_model.predict(X_eval)
y_eval_prob = best_model.predict_proba(X_eval)[:, 1]
eval_classification_report = classification_report(y_eval, y_eval_pred)
eval_roc_auc = roc_auc_score(y_eval, y_eval_prob)
print("\nEvaluation Classification Report:\n", eval_classification_report)
print(f"Evaluation ROC-AUC Score: {eval_roc_auc:.4f}")

# Save the Model
joblib.dump(best_model, 'xgb_fraud_detection_model2.pkl')
print("Model saved as 'xgb_fraud_detection_model2.pkl'.")
