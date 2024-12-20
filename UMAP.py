#%%
import pandas as pd
import umap
import numpy as np
import hdbscan
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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

# %% Data pre-processing of dati_parametri_al_31122020

# Load the data
dati_parametri_al_31122020=pd.read_csv('./data/dati_parametri_al_31122020.csv')

#Remove Outliers with no physical/medical meaning

#Weight(Kg)
dati_parametri_al_31122020.loc[dati_parametri_al_31122020['Weight(Kg)']>200,'Weight(Kg)']=np.nan
dati_parametri_al_31122020.loc[dati_parametri_al_31122020['Weight(Kg)']<5,'Weight(Kg)']=np.nan

#Height(cm)
dati_parametri_al_31122020.loc[dati_parametri_al_31122020['Height(cm)']>250,'Height(cm)']=np.nan
dati_parametri_al_31122020.loc[dati_parametri_al_31122020['Height(cm)']<30,'Height(cm)']=np.nan

#Correction on inverted values for max and min blood pressure
condition = dati_parametri_al_31122020['BloodPressureMax'] < dati_parametri_al_31122020['BloodPressureMin']
dati_parametri_al_31122020.loc[condition, ['BloodPressureMax', 'BloodPressureMin']] = dati_parametri_al_31122020.loc[condition, ['BloodPressureMin', 'BloodPressureMax']].values

#BloodPressureMax <20 removal (impossible to have max blood pressure lower than 20)
dati_parametri_al_31122020.loc[dati_parametri_al_31122020['BloodPressureMax']<20,'BloodPressureMax']=np.nan

#BloodPressureMin <20 removal (impossible to have min blood pressure lower than 20)
dati_parametri_al_31122020.loc[dati_parametri_al_31122020['BloodPressureMin']<20,'BloodPressureMin']=np.nan

#OxygenSaturation(%)<50 removal (saturation lower than 50 not feasible for life)
dati_parametri_al_31122020.loc[dati_parametri_al_31122020['OxygenSaturation']<50,'OxygenSaturation']=np.nan

#HeartRate>300 and HeartRate<15 removal (no life)
dati_parametri_al_31122020.loc[dati_parametri_al_31122020['HeartRate']>300,'HeartRate']=np.nan
dati_parametri_al_31122020.loc[dati_parametri_al_31122020['HeartRate']<15,'HeartRate']=np.nan

#BMI>100 and BMI<10 removal
dati_parametri_al_31122020.loc[dati_parametri_al_31122020['BMI']>100,'BMI']=np.nan      
dati_parametri_al_31122020.loc[dati_parametri_al_31122020['BMI']<10,'BMI']=np.nan

#RespiratoryRate=0 removal
dati_parametri_al_31122020.loc[dati_parametri_al_31122020['RespiratoryRate']<1,'RespiratoryRate']=np.nan

dati_parametri_al_31122020.drop(columns=['CenterId','CenterName','Date'], inplace=True)
dati_parametri_al_31122020 = dati_parametri_al_31122020.pivot_table(index= 'PatientId', aggfunc='mean')


#%%
# Read the CSV file
dati_esami = pd.read_csv('./data/dati_esami_al_31122020.csv')

# Filter the dataframe to remove unnecessary columns (excluded Note, GroupName,GroupId,DiagnosisId)
dati_esami=dati_esami[["DiagnosisName","ExamName","Outcome","PatientId","Date","ExamId"]]

# Remove rows with missing Outcome value (clearly not useful for the analysis)
dati_esami=dati_esami.dropna(subset=["Outcome"], inplace=False, ignore_index=True)

# Change decimal separator from comma to dot, to avoid errors when using pd.to_numeric
dati_esami["Outcome"]=dati_esami["Outcome"].str.replace(',', '.') #replace comma with dot 

# Convert to numeric the Outcome column 
#NOTE: errors='coerce' is used to force conversion of non-numeric values to NaN
dati_esami["Outcome"]=pd.to_numeric(dati_esami["Outcome"], errors='coerce')

# For each patient, for each diagnosis, calculate the mean of the outcomes
dati_esami = dati_esami.pivot_table(index=['DiagnosisName', 'PatientId'], columns='ExamName', values='Outcome', aggfunc='mean')
dati_esami = dati_esami.reset_index(drop=False)


#%%
# Merge the DataFrames on the 'PatientId' column
merged_df = pd.merge(WM_df, dati_esami, on='PatientId', how='inner')

# Merge the resulting DataFrame with the 'dati_parametri_al_31122020' DataFrame, to add parameters column
merged_df = pd.merge(merged_df, dati_parametri_al_31122020, on='PatientId', how='left')


#NOTE: to treat missing values, we can drop columns with less than 30 non-missing values,
#  and fill the remaining missing values with the mean of the column, so that it contributes less to the analysis,
#  after the application of the StandardScaler() function
merged_df=merged_df.dropna(axis=1,thresh=30,ignore_index=True)
merged_df = merged_df.fillna(merged_df.drop(columns=['PatientId','Infarto silente','DiagnosisName']).mean())



#%%
# UMAP for dimensionality reduction and visualization
features = merged_df.drop(columns=['PatientId', 'DiagnosisName', 'Infarto silente']).copy()
labels = merged_df['Infarto silente'].copy()

#Scale the features
features_scaled=StandardScaler().fit_transform(features)

# Apply UMAP
reducer = umap.UMAP(n_components=2, n_neighbors=5, init='pca', metric="euclidean", learning_rate=0.5, n_epochs=500,random_state=24)
umap_result = reducer.fit_transform(features_scaled)

# Create a DataFrame for the UMAP results
umap_df = pd.DataFrame(umap_result, columns=['UMAP1', 'UMAP2'])
umap_df['Infarto silente'] = labels

# Create HDBSCAN clusterer
clusterer = hdbscan.HDBSCAN(min_cluster_size=5)

# Fit HDBSCAN to the UMAP results
cluster_df=umap_df.copy()
cluster_df['Cluster'] = clusterer.fit_predict(umap_df.drop(columns='Infarto silente'))

# Calculate the percentage of each classification within each cluster
classification_percentages = []

# For each cluster, compute the percentage of each class and the size of the cluster
for cluster in cluster_df['Cluster'].unique():
    if cluster == -1:
        print('Cluster -1 contains outliers')
    cluster_data = cluster_df[cluster_df['Cluster'] == cluster]
    total_count = len(cluster_data)
    class_1_count = len(cluster_data[cluster_data['Infarto silente'] == "Sano"])
    class_0_count = len(cluster_data[cluster_data['Infarto silente'] == 'Malato'])
    
    class_1_percentage = (class_1_count / total_count) * 100
    class_0_percentage = (class_0_count / total_count) * 100

    classification_percentages.append({
        'Cluster': cluster,
        'Class "Sano" Percentage': class_1_percentage,
        'Class "Malato" Percentage': class_0_percentage,
        'Cluster Size': total_count
    })

percentages_df = pd.DataFrame(classification_percentages)
print(percentages_df)

# Write the results to a text file in the summary folder
with open('summary/clustering.txt', 'w') as file:
    file.write(percentages_df.to_string())

#NOTE: I do know that classification percentages are not the best way to evaluate the clustering,
#  but it is a good starting point to understand the results


#%% Visualize the results of UMAP and HDBSCAN

# Plot the UMAP results
# 2D plot
plt.figure(figsize=(10, 6))
for label in umap_df['Infarto silente'].unique():
    subset = umap_df[umap_df['Infarto silente'] == label]
    plt.scatter(subset['UMAP1'], subset['UMAP2'], label=label)
plt.legend()
plt.title('UMAP 2D')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
#plt.savefig('./plots/umap_2D.png')


# Plot the clustering results
# 2D plot
plt.figure(figsize=(10, 6))
for label in cluster_df['Cluster'].unique():
    subset = cluster_df[cluster_df['Cluster'] == label]
    plt.scatter(subset['UMAP1'], subset['UMAP2'], label=label)
plt.legend()
plt.title('HDBSCAN Clustering')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
#plt.savefig('./plots/clustering_2D.png')



#%% PERFORMING UMAP WITH VARIABLES SCALED TO [0,1] RANGE, TO SEE IF RESULTS IMPROVES/WORSENS

# UMAP for dimensionality reduction and visualization
features = merged_df.drop(columns=['PatientId', 'DiagnosisName', 'Infarto silente']).copy()
labels = merged_df['Infarto silente'].copy()

# Rescale the features to [0, 1] range
min_max_scaler = MinMaxScaler()
features_scaled = min_max_scaler.fit_transform(features)

# Apply UMAP
reducer = umap.UMAP(n_components=2, n_neighbors=5, init='pca', metric="euclidean", learning_rate=0.5, n_epochs=500,random_state=24)
umap_result = reducer.fit_transform(features_scaled)

# Create a DataFrame for the UMAP results
umap_df = pd.DataFrame(umap_result, columns=['UMAP1', 'UMAP2'])
umap_df['Infarto silente'] = labels

# Plot the UMAP results
# 2D plot
plt.figure(figsize=(10, 6))
for label in umap_df['Infarto silente'].unique():
    subset = umap_df[umap_df['Infarto silente'] == label]
    plt.scatter(subset['UMAP1'], subset['UMAP2'], label=label)
plt.legend()
plt.title('UMAP 2D scaled to [0, 1]')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
#plt.savefig('./plots/umap_2D_scaled_0_1.png')

