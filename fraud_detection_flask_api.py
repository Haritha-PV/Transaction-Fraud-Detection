from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('xgb_fraud_detection_model2.pkl')

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML page

@app.route('/predict', methods=['POST'])
def predict_fraud():
    try:
        # Get form data
        data = {
            'step': request.form.get('step', type=int),
            'amount': request.form.get('amount', type=float),
            'oldbalanceOrg': request.form.get('oldbalanceOrg', type=float),
            'newbalanceOrig': request.form.get('newbalanceOrig', type=float),
            'oldbalanceDest': request.form.get('oldbalanceDest', type=float),
            'newbalanceDest': request.form.get('newbalanceDest', type=float),
            'type': request.form.get('type'),
        }

        # Validate input
        if None in data.values():
            return render_template('index.html', error="All fields must be filled.")

        # Create DataFrame from input
        input_df = pd.DataFrame([data])

        # Drop unnecessary columns
        input_df = input_df.drop(columns=['step', 'newbalanceOrig', 'newbalanceDest'], errors='ignore')

        # One-hot encode the 'type' column
        input_df = pd.get_dummies(input_df, columns=['type'], prefix='type', drop_first=True)

        # Ensure all required columns are present
        required_features = model.feature_names_in_  # Features used during training
        for feature in required_features:
            if feature not in input_df.columns:
                input_df[feature] = 0  # Add missing features with default values

        # Predict fraud
        prediction = model.predict(input_df)[0]
        confidence_score = model.predict_proba(input_df)[:, 1][0]  # Probability of fraud

        # Prepare result
        result = {
            'Input Details': data,
            'Prediction Result': 'Fraud Detected' if prediction == 1 else 'No Fraud',
            'Confidence Score': f"{confidence_score:.2f}"
        }

        return render_template('index.html', result=result)  # Pass result to the template

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)