#%%
import pandas as pd
import umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

#%%
# Read the Excel file
WM_df = pd.read_excel('.\data\RMN+angioRMN 2021 Luglio Genomed4ALL.xlsx')

# Remove two empty columns
WM_df = WM_df.drop(columns=['Unnamed: 2', 'Unnamed: 1'])

# Change the id column to match the other dataframes
WM_df = WM_df.rename(columns={'id': 'PatientId'})
WM_df['PatientId'] = WM_df['PatientId'].astype(str)

# Add the correct number of zeros and the prefix 'PD' to the PatientId column
WM_df['PatientId'] = WM_df['PatientId'].str.zfill(4)
WM_df['PatientId'] = 'PD' + WM_df['PatientId']

# Convert the columns "esito" and "RMN Cerebrale > Infarto silente  (sì/no)" to boolean, in order to perform further analysis
WM_df=WM_df.rename(columns={'RMN Cerebrale > Infarto silente  (sì/no)': 'Infarto silente'})

# Keep only the columns "PatientId" and "Infarto silente"
WM_df = WM_df[['PatientId', 'Infarto silente']]

# Remove duplicates, keeping the last occurrence (which will be 'SI' if the patient has been diagnosed a silent infarct at least once)
WM_df = WM_df.sort_values(by='Infarto silente')
WM_df = WM_df.drop_duplicates(subset='PatientId', keep='last',ignore_index=True)

# Change the values of the 'Infarto silente' column to 'Malato' and 'Sano', for easier interpretation
WM_df['Infarto silente'] = WM_df['Infarto silente'].map({'SI': 'Malato', 'NO': 'Sano'})


#%%
# Read the CSV file
dati_esami = pd.read_csv('./data/dati_esami_al_31122020.csv')

#Filter the dataframe to remove unnecessary columns (excluded Note, GroupName,GroupId,DiagnosisId)
dati_esami=dati_esami[["DiagnosisName","ExamName","Outcome","PatientId","Date","ExamId"]]

#Remove rows with missing Outcome value (clearly not useful for the analysis)
dati_esami=dati_esami.dropna(subset=["Outcome"], inplace=False, ignore_index=True)

#Change decimal separator from comma to dot, to avoid errors when using pd.to_numeric
dati_esami["Outcome"]=dati_esami["Outcome"].str.replace(',', '.') #replace comma with dot 

#Convert to numeric the Outcome column 
#NOTE: errors='coerce' is used to force conversion of non-numeric values to NaN
dati_esami["Outcome"]=pd.to_numeric(dati_esami["Outcome"], errors='coerce')

#For each patient, for each diagnosis, calculate the mean of the outcomes
dati_esami = dati_esami.pivot_table(index=['DiagnosisName', 'PatientId'], columns='ExamName', values='Outcome', aggfunc='mean')
dati_esami = dati_esami.reset_index(drop=False)


#%%
# Merge the DataFrames on the 'PatientId' column
merged_df = pd.merge(WM_df, dati_esami, on='PatientId', how='inner')


#NOTE: to treat missing values, we can drop columns with less than 30 non-missing values, and fill the remaining missing values with 0
merged_df=merged_df.dropna(axis=1,thresh=30,ignore_index=True)
merged_df=merged_df.fillna(0)
