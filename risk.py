import pandas as pd
# Load the dataset
data = pd.read_csv('outreupdated_dataset.csv')
# Define a function to calculate risk
def calculate_risk(row):
    return (row['Likelihood'] * row['Asset_Value']) - row['Mitigated_Risk'] + row['Uncertainty']

# Apply the function to each row to calculate risk
data['Risk'] = data.apply(calculate_risk, axis=1)

# Calculate maximum risk
max_risk = data['Risk'].max()

# Convert risk to percentage
data['Risk Percentage'] = (data['Risk'] / max_risk) * 100

# Define a function to map risk percentages to risk levels
def map_risk_level(risk_percentage):
    if risk_percentage < 25:
        return 'Low Risk'
    elif 25 <= risk_percentage < 60:
        return 'Moderate Risk'
    elif 60 <= risk_percentage < 85:
        return 'High Risk'
    else:
        return 'Very High Risk'

# Create a new column for risk levels based on risk percentages
data['Risk Level'] = data['Risk Percentage'].apply(map_risk_level)

# Save the updated dataset to a new CSV file
data.to_csv('out_risk_updated_dataset.csv', index=False)