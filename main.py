import pandas as pd

# Load the dataset
data = pd.read_csv('MOCK_DATA (2).csv')
data['Previous_Incidents'] = data['Previous_Incidents'].str.extract(r'(\d+)')

# Convert the column to numeric data type
data['Previous_Incidents'] = pd.to_numeric(data['Previous_Incidents'], errors='coerce')

# Calculate Vulnerability
data['Vulnerability'] = data['Compliance_Status'] * data['Cybersecurity_Score'] + data['Physical_Security_Score'] * 0.5 + data['Vendor_Risk_Score'] * 0.3

# Calculate Likelihood of Risk Occurrence
data['Likelihood'] = data['Environmental_Risk'] * 0.8 + data['Previous_Incidents'] * 0.2

# Calculate Uncertainty of Risk
data['Uncertainty'] = data['Age'] * 0.5 + data['Maintenance_Cost'] * 0.3 + data['Compliance_Status'] * 0.2
#calculate mitigated_risk
data['Mitigated_Risk'] = (data['Likelihood'] * data['Asset_Value']) * (1 - (data['Physical_Security_Score'] + data['Vendor_Risk_Score']) / 200)



# Save the updated dataset to a new CSV file
data.to_csv('outreupdated_dataset.csv', index=False)
