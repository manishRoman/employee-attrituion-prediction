import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. Load your dataset
df = pd.read_csv("project data science/my_IBM_cleaned_dataset.csv")

# 2. Select ONLY numeric columns and drop the Target variable ('Attrition')
# VIF cannot calculate math on text, and we only test the independent variables
X_vif = df.select_dtypes(include=['number']).drop('Attrition', axis=1, errors='ignore')

# 3. Create a DataFrame to hold the VIF results
vif_data = pd.DataFrame()
vif_data["Feature"] = X_vif.columns

# 4. Calculate VIF for each feature
# This loops through every column and calculates how much it overlaps with the others
vif_data["VIF_Score"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]

# 5. Sort the results to see the worst offenders at the top
vif_data = vif_data.sort_values(by="VIF_Score", ascending=False)

print("VARIANCE INFLATION FACTOR (VIF) SCORES:")
print("-" * 45)
print(vif_data.to_string(index=False))
print("-" * 45)