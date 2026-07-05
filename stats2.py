import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency

# 1. Load your dataset
# Replace 'your_dataset.csv' with your actual IBM HR dataset filename
df = pd.read_csv(r"C:\Users\mr006\Downloads\ibm_hr_model_ready.xls")

# Ensure Attrition is numeric (1 for Yes, 0 for No) if it isn't already
if df['Attrition'].dtype == 'O':
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Split data into attrited and retained for T-tests
attrited = df[df['Attrition'] == 1]
retained = df[df['Attrition'] == 0]

# 2. Identify Categorical vs. Continuous columns
# Exclude the target variable ('Attrition') and purely identifier columns (like 'EmployeeCount' or 'EmployeeNumber')
columns_to_exclude = ['Attrition', 'EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
feature_cols = [col for col in df.columns if col not in columns_to_exclude]

# A common way to separate categorical vs continuous in the IBM dataset
# Variables with 'object' dtype are categorical. Some numeric columns (like EducationLevel) might be categorical, 
# but for basic baseline filtering, we will split by pandas dtype and unique values.
categorical_cols = [col for col in feature_cols if df[col].dtype == 'object' or df[col].nunique() <= 10]
continuous_cols = [col for col in feature_cols if col not in categorical_cols]

# Lists to keep track of what stays and what goes
significant_features = []
dropped_features = []

print("="*50)
print("AUTOMATED FEATURE SELECTION RUNNING...")
print("="*50)

# 3. Test Continuous Features (Independent T-Test)
for col in continuous_cols:
    stat, p_value = ttest_ind(attrited[col], retained[col], nan_policy='omit')
    if p_value < 0.05:
        significant_features.append(col)
    else:
        dropped_features.append(col)

# 4. Test Categorical Features (Chi-Square Test)
for col in categorical_cols:
    contingency_table = pd.crosstab(df[col], df['Attrition'])
    stat, p_value, dof, expected = chi2_contingency(contingency_table)
    if p_value < 0.05:
        significant_features.append(col)
    else:
        dropped_features.append(col)

# 5. Review Results
print(f"\n[KEPT] {len(significant_features)} Significant Features (p < 0.05):")
print(significant_features)

print(f"\n[DROPPED] {len(dropped_features)} Noisy Features (p >= 0.05):")
print(dropped_features)

# 6. Create the final clean dataframe for your Logistic Regression model
# We keep the significant features plus the target variable 'Attrition'
final_columns_to_keep = significant_features + ['Attrition']
df_clean = df[final_columns_to_keep]

print("\n" + "="*50)
print(f"Original Dataset Shape: {df.shape}")
print(f"Cleaned Dataset Shape:  {df_clean.shape}")
print("="*50)

# You can now save this clean dataset or immediately pass df_clean into your Logistic Regression pipeline.
# df_clean.to_csv('ibm_hr_cleaned.csv', index=False)