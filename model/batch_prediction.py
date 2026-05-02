import pandas as pd
import joblib

model = joblib.load("model/churn_model.pkl")
columns = joblib.load("model/columns.pkl")

# Create all columns = 0
input_data = {col: 0 for col in columns}

# INPUT VALUES
tenure = 4
monthly = 79
total = 399
gender = "Female"
contract = "Month-to-month"

# Fill numeric
input_data['Tenure_Months'] = tenure
input_data['Monthly_Charges'] = monthly
input_data['Total_Charges'] = total

# Gender encoding
if gender == "Male":
    input_data['Gender_Male'] = 1

# Contract encoding
if contract == "One year":
    input_data['Contract_One year'] = 1
elif contract == "Two year":
    input_data['Contract_Two year'] = 1
# Month-to-month = default (all 0)

# Convert to dataframe
df = pd.DataFrame([input_data])

# Predict
df["Churn_Prob"] = model.predict_proba(df)[:,1]

# 🔥 ADD READABLE COLUMNS (IMPORTANT)
df["Gender"] = gender
df["Contract"] = contract

# Save
df.to_csv("live_prediction.csv", index=False)

print("✅ Prediction saved")