#%%
import pandas as pd
import umap
import hdbscan
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

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


#NOTE: to treat missing values, we can drop columns with less than 30 non-missing values, and fill the remaining missing values with 0
merged_df=merged_df.dropna(axis=1,thresh=30,ignore_index=True)
merged_df=merged_df.fillna(0)


#%%
# UMAP for dimensionality reduction and visualization
features = merged_df.drop(columns=['PatientId', 'DiagnosisName', 'Infarto silente']).copy()
labels = merged_df['Infarto silente'].copy()

# Apply UMAP
reducer = umap.UMAP(n_components=3, n_neighbors=10, init='pca', metric="cosine", learning_rate=0.5, n_epochs=2000)
umap_result = reducer.fit_transform(features)

# Create a DataFrame for the UMAP results
umap_df = pd.DataFrame(umap_result, columns=['UMAP1', 'UMAP2', 'UMAP3'])
umap_df['Infarto silente'] = labels

# Create HDBSCAN clusterer
clusterer = hdbscan.HDBSCAN(min_cluster_size=5)

# Fit HDBSCAN to the UMAP results
cluster_df=umap_df.copy()
cluster_df['Cluster'] = clusterer.fit_predict(umap_df.drop(columns='Infarto silente'))

# Calculate the percentage of each classification within each cluster
classification_percentages = []

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
plt.title('UMAP 2D Projection')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
#plt.savefig('./plots/umap_2D.png')

# 3D plot
fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111, projection='3d')
for label in umap_df['Infarto silente'].unique():
    subset = umap_df[umap_df['Infarto silente'] == label]
    ax.scatter(subset['UMAP1'], subset['UMAP2'], subset['UMAP3'], label=label)
ax.legend()
ax.set_title('UMAP 3D Projection')
ax.set_xlabel('UMAP1')
ax.set_ylabel('UMAP2')
ax.set_zlabel('UMAP3')
fig.subplots_adjust(left=0, right=1, top=0.95, bottom=0.01)
#plt.savefig('./plots/umap_3D.png')


# Plot the clustering results
# 2D plot
plt.figure(figsize=(10, 6))
for label in cluster_df['Cluster'].unique():
    subset = cluster_df[cluster_df['Cluster'] == label]
    plt.scatter(subset['UMAP1'], subset['UMAP2'], label=label)
plt.legend()
plt.title('Clustering results')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
#plt.savefig('./plots/clustering_2D.png')

# 3D plot
fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111, projection='3d')
for label in cluster_df['Cluster'].unique():
    subset = cluster_df[cluster_df['Cluster'] == label]
    ax.scatter(subset['UMAP1'], subset['UMAP2'], subset['UMAP3'], label=label)
ax.legend()
ax.set_title('3D clustering results') 
ax.set_xlabel('UMAP1')
ax.set_ylabel('UMAP2')
ax.set_zlabel('UMAP3')
fig.subplots_adjust(left=0, right=1, top=0.95, bottom=0.01)
#plt.savefig('./plots/clustering_3D.png')



#%% PERFORMING UMAP WITH VARIABLES SCALED TO [0,1] RANGE, TO SEE IF RESULTS IMPROVES/WORSENS

# UMAP for dimensionality reduction and visualization
features = merged_df.drop(columns=['PatientId', 'DiagnosisName', 'Infarto silente']).copy()
labels = merged_df['Infarto silente'].copy()

# Rescale the features to [0, 1] range
min_max_scaler = MinMaxScaler()
features_scaled = min_max_scaler.fit_transform(features)

# Apply UMAP
reducer = umap.UMAP(n_components=3, n_neighbors=10, init='pca', metric="cosine", learning_rate=0.5, n_epochs=2000)
umap_result = reducer.fit_transform(features_scaled)

# Create a DataFrame for the UMAP results
umap_df = pd.DataFrame(umap_result, columns=['UMAP1', 'UMAP2', 'UMAP3'])
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

# 3D plot
fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111, projection='3d')
for label in umap_df['Infarto silente'].unique():
    subset = umap_df[umap_df['Infarto silente'] == label]
    ax.scatter(subset['UMAP1'], subset['UMAP2'], subset['UMAP3'], label=label)
ax.legend()
ax.set_title('UMAP 3D scaled to [0, 1]')
ax.set_xlabel('UMAP1')
ax.set_ylabel('UMAP2')
ax.set_zlabel('UMAP3')
fig.subplots_adjust(left=0, right=1, top=0.95, bottom=0.01)
#plt.savefig('./plots/umap_3D_scaled_0_1.png')
