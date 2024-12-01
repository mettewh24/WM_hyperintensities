#%% Load the necessary libraries
import pandas as pd
from scipy.stats import chi2_contingency


# %% Preprocess the data related to white matter hyperintensities (i.e. silent infarcts)

# Read the Excel file with the data about the silent infarcts
WM_df = pd.read_excel('.\data\RMN+angioRMN 2021 Luglio Genomed4ALL.xlsx')

# Remove two empty columns
WM_df = WM_df.drop(columns=['Unnamed: 2', 'Unnamed: 1'])

# Change the id column to match the other dataframes
WM_df = WM_df.rename(columns={'id': 'PatientId'})
WM_df['PatientId'] = WM_df['PatientId'].astype(str)

# Add the correct number of 0 and the prefix 'PD' to the PatientId column
WM_df['PatientId'] = WM_df['PatientId'].str.zfill(4)
WM_df['PatientId'] = 'PD' + WM_df['PatientId']

#Rename the column "RMN Cerebrale > Infarto silente  (sì/no)" to "Infarto silente", just for brevity
WM_df=WM_df.rename(columns={'RMN Cerebrale > Infarto silente  (sì/no)': 'Infarto silente'})
WM_df['Infarto silente'] = WM_df['Infarto silente'].map({'SI': True, 'NO': False})

# Keep only the columns "PatientId", "Infarto silente"
WM_df = WM_df[['PatientId', 'Infarto silente']]

# Remove duplicates (i.e. patients with multiple entries), keeping the last occurrence (which will be 'True' if it exists)
WM_df = WM_df.sort_values(by='Infarto silente')
WM_df = WM_df.drop_duplicates(subset='PatientId', keep='last',ignore_index=True)


# %%
# Load the data containing the diagnosis of the patients
dati_esami=pd.read_csv('./data/dati_esami_al_31122020.csv')

#Filter the dataframe to keep only the columns "PatientId" and "DiagnosisName"
dati_esami=dati_esami[["PatientId","DiagnosisName"]]
dati_esami=dati_esami.drop_duplicates(ignore_index=True)

#%%
# Merge the two dataframes on "PatientId" and create a contingency table
merged_df = pd.merge(dati_esami,WM_df, on='PatientId', how='inner')
contingency_table = pd.crosstab(merged_df['DiagnosisName'],merged_df['Infarto silente'])

#Chi squared test on contingency table 
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi2: {chi2}, P-value: {p}, Degrees of Freedom: {dof}")


# Write the results to a text file in the summary folder
with open('summary/chi2_test_results.txt', 'w') as file:
    file.write(f"Chi2: {chi2}\n")
    file.write(f"P-value: {p}\n")
    file.write(f"Degrees of Freedom: {dof}\n")
