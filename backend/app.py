from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd   # 🔥 added

app = Flask(__name__)
CORS(app)

# Load model + columns
model = joblib.load("model/churn_model.pkl")
columns = joblib.load("model/columns.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Step 1: Initialize all features to 0
        input_data = {col: 0 for col in columns}
        
        # Step 2: Map input fields
        input_data['Tenure_Months'] = data.get('Tenure_Months', 0)
        input_data['Monthly_Charges'] = data.get('Monthly_Charges', 0)
        input_data['Total_Charges'] = data.get('Total_Charges', 0)
        
        # Step 3: Gender encoding
        gender = data.get('gender', '')
        if gender == "Male":
            input_data['Gender_Male'] = 1
        
        # Step 4: Contract encoding
        contract = data.get('contract', '')
        if contract == "One year":
            input_data['Contract_One year'] = 1
        elif contract == "Two year":
            input_data['Contract_Two year'] = 1
        # Month-to-month = default
        
        # Step 5: Convert to model input
        features = np.array([input_data[col] for col in columns]).reshape(1, -1)
        
        # Step 6: Prediction
        prob = model.predict_proba(features)[0][1]
        churn = 1 if prob > 0.3 else 0
        
        # 🔥 Step 7: Create DataFrame for CSV
        df = pd.DataFrame([input_data])
        
        # Add prediction columns
        df["Churn_Prob"] = prob
        df["Prediction"] = churn
        df["Prediction_Label"] = "Will Churn" if churn == 1 else "Will Not Churn"
        
        # 🔥 Step 8: Save to CSV (Power BI reads this)
        df.to_csv("live_prediction.csv", index=False)
        
        # Step 9: Return API response
        return jsonify({
            "churn": churn,
            "churn_probability": round(float(prob), 4),
            "prediction": df["Prediction_Label"].iloc[0]
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "Check input fields"
        }), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)