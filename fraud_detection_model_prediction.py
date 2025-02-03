
import pandas as pd
import sqlite3
import joblib

# Load the trained model and scaler
model = joblib.load('xgb_fraud_detection_model.pkl')

# Connect to the SQLite database and read the data
conn = sqlite3.connect('Database.db')
df = pd.read_sql_query('SELECT * FROM Fraud_detection', conn)

# Preprocessing steps
numeric_columns = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                   'oldbalanceDest', 'newbalanceDest', 'isFraud', 'isFlaggedFraud']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = pd.get_dummies(df, columns=['type'], drop_first=True)

df['newbalanceOrig'] = df['newbalanceOrig'].fillna(0)
df['newbalanceDest'] = df['newbalanceDest'].fillna(0)

# Create new features
df['balanceDifference'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['destinationBalanceDiff'] = df['oldbalanceDest'] - df['newbalanceDest']

df['net_transaction'] = df['amount'] - (df['balanceDifference'] - df['destinationBalanceDiff'])

# Transaction Ratios:Calculate the proportion of the transferred amount to the origin and destination balances.
df['origin_amount_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)  # Adding 1 to avoid division by zero
df['dest_amount_ratio'] = df['amount'] / (df['oldbalanceDest'] + 1)

# Drop unnecessary columns
df.drop(columns=['nameOrig', 'nameDest'], inplace=True)

# Drop one feature from each pair
features_to_drop = ['newbalanceOrig', 'newbalanceDest', 'origin_amount_ratio','step']  # Example
df = df.drop(features_to_drop, axis=1)

# Make predictions
X_pred = df.drop('isFraud', axis=1, errors='ignore')  # Avoid error if 'isFraud' not present
predictions = model.predict(X_pred)

# Append predictions to the dataset
df['Fraud_Prediction'] = predictions

# Display results
print("\n--- Prediction Summary ---")
fraudulent_count = sum(df['Fraud_Prediction'])
total_count = len(df)
print(f"Total Transactions Processed: {total_count}")
print(f"Fraudulent Transactions Detected: {fraudulent_count}")
print(f"Non-Fraudulent Transactions: {total_count - fraudulent_count}")

# Show first few rows with predictions
print("\n--- Sample Predictions ---")
print(df[['amount', 'type_TRANSFER', 'type_CASH_OUT', 'Fraud_Prediction']].head())

# Save or output results
df.to_csv('fraud_predictions.csv', index=False)
print("Predictions saved to 'fraud_predictions.csv'.")