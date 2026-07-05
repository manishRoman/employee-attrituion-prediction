import pandas as pd
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

print("="*50)
print("1. LOADING DATA & CREATING BASELINE")
print("="*50)

# 1. Load the cleaned dataset (the one without noise/multicollinearity)
df = pd.read_csv(r"C:\Users\mr006\Downloads\ibm_hr_model_ready.xls")

# 2. Separate Features (X) and Target (y)
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# 3. Generate the "Average Employee" Baseline Profile
baseline_profile = {}
for col in X.columns:
    if X[col].dtype == 'object':
        baseline_profile[col] = str(X[col].mode()[0])
    else:
        baseline_profile[col] = float(X[col].mean())

# Save the baseline profile as a JSON file for the web app
with open('baseline_employee.json', 'w') as file:
    json.dump(baseline_profile, file, indent=4)
print("Success: 'baseline_employee.json' saved.")

print("\n" + "="*50)
print("2. PREPROCESSING & DUMMY VARIABLES")
print("="*50)

# 4. Convert text categories into numeric dummy variables
X_encoded = pd.get_dummies(X, drop_first=True)

# 5. Save the exact column structure
# The web app MUST know the exact order and names of these dummy columns to make a prediction
model_columns = list(X_encoded.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Success: 'model_columns.pkl' saved.")

print("\n" + "="*50)
print("3. TRAINING & SCALING")
print("="*50)

# 6. Split the data (80% for training, 20% for testing the accuracy)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 7. Scale the features (Crucial for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler so the web app can scale the stakeholder's input
joblib.dump(scaler, 'scaler.pkl')
print("Success: 'scaler.pkl' saved.")

# 8. Initialize and Train the Logistic Regression Model
# max_iter=1000 ensures the math has enough time to converge
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train_scaled, y_train)
print("Success: Logistic Regression model trained.")

print("\n" + "="*50)
print("4. MODEL EVALUATION")
print("="*50)

# 9. Test the model's accuracy on the unseen 20% of data
y_pred = log_model.predict(X_test_scaled)
print("Classification Report on Test Data:\n")
print(classification_report(y_test, y_pred))

print("\n" + "="*50)
print("5. EXPORTING FINAL MODEL")
print("="*50)

# 10. Save the actual trained model
joblib.dump(log_model, 'logistic_model.pkl')
print("Success: 'logistic_model.pkl' saved.")
print("\nALL FILES READY FOR DEPLOYMENT!")