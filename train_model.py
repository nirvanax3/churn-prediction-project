import pandas as pd
import joblib

# =========================
# LOAD DATA
# =========================
df = pd.read_excel("Telco_customer_churn.xlsx")

# =========================
# CLEANING
# =========================
df.drop([
    'CustomerID','Count','Country','State','City','Zip Code',
    'Lat Long','Latitude','Longitude',
    'Churn Label','Churn Score','CLTV','Churn Reason'
], axis=1, inplace=True)

# Rename target column
df.rename(columns={'Churn Value': 'Churn'}, inplace=True)

# =========================
# FIX DATA TYPES
# =========================
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')

# Fill missing numeric values
df['Total Charges'] = df['Total Charges'].fillna(df['Total Charges'].median())

# Clean column names
df.columns = df.columns.str.replace(' ', '_')

# =========================
# ENCODING
# =========================

# Binary encoding
binary_cols = ['Partner', 'Dependents', 'Phone_Service', 'Paperless_Billing']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# Senior Citizen encoding
df['Senior_Citizen'] = df['Senior_Citizen'].map({'Yes': 1, 'No': 0})

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# =========================
# FINAL NULL CHECK + FIX
# =========================
print("\n🔍 Null values per column:\n")
print(df.isnull().sum())

# Fill any remaining nulls (safety)
df = df.fillna(0)

print("\n✅ Total Null values after fix:", df.isnull().sum().sum())
print("📊 Dataset Shape:", df.shape)

# =========================
# SPLIT DATA
# =========================
from sklearn.model_selection import train_test_split

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MODEL TRAINING
# =========================
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)

model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
from sklearn.metrics import classification_report

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob > 0.3).astype(int)

print("\n📈 Classification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# FEATURE IMPORTANCE
# =========================
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance = feature_importance.sort_values(ascending=False)

print("\n🔥 Top 10 Important Features:\n")
print(feature_importance.head(10))

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, "churn_model.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")

# =========================
# SAVE CLEAN DATA (POWER BI)
# =========================
df.to_csv("cleaned_churn.csv", index=False)

print("\n✅ Model & dataset saved successfully!")