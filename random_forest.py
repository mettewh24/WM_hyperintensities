#%%
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


#%%#%%
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

# Rename the column 'RMN Cerebrale > Infarto silente  (sì/no)' to 'Infarto silente', for simplicity
WM_df=WM_df.rename(columns={'RMN Cerebrale > Infarto silente  (sì/no)': 'Infarto silente'})

# Keep only the columns "PatientId" and "Infarto silente"
WM_df = WM_df[['PatientId', 'Infarto silente']]

# Remove duplicates, keeping the last occurrence (which will be 'SI' if the patient has been diagnosed a silent infarct at least once)
WM_df = WM_df.sort_values(by='Infarto silente')
WM_df = WM_df.drop_duplicates(subset='PatientId', keep='last',ignore_index=True)



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
# NOTE: errors='coerce' is used to force conversion of non-numeric values to NaN
dati_esami["Outcome"]=pd.to_numeric(dati_esami["Outcome"], errors='coerce')

# Perform a pivot table to have the ExamName values as columns, keeping the PatientId and DiagnosisName columns, and the mean Outcome values as the table values
dati_esami = dati_esami.pivot_table(index=['DiagnosisName', 'PatientId'], columns='ExamName', values='Outcome', aggfunc='mean')
dati_esami = dati_esami.reset_index(drop=False)



#%%
# Merge the DataFrames on the 'PatientId' column
merged_df = pd.merge(WM_df, dati_esami, on='PatientId', how='inner')

#NOTE: column with less than 50 non-null values are dropped, all the other Nan are filled with 0
merged_df=merged_df.dropna(axis=1,thresh=50,ignore_index=True)
merged_df.fillna(0, inplace=True)

# Divide the dataset into two subsets: one with patients diagnosed, and one with patients not diagnosed
df_SI = merged_df[merged_df['Infarto silente'] == 'SI']
df_NO = merged_df[merged_df['Infarto silente'] == 'NO']

# Split the dataset into training and test sets, separately for the two subsets, to allow different sizes of sick and healty patients
SI_train, SI_test = train_test_split(df_SI, train_size=0.6) 
NO_train, NO_test = train_test_split(df_NO, train_size=0.9)

# Merge the training and test sets
X_train= pd.concat([SI_train.drop(columns=['PatientId', 'DiagnosisName', 'Infarto silente']), NO_train.drop(columns=['PatientId', 'DiagnosisName', 'Infarto silente'])])
X_test= pd.concat([SI_test.drop(columns=['PatientId', 'DiagnosisName', 'Infarto silente']), NO_test.drop(columns=['PatientId', 'DiagnosisName', 'Infarto silente'])])
y_train= pd.concat([SI_train['Infarto silente'], NO_train['Infarto silente']])
y_test= pd.concat([SI_test['Infarto silente'], NO_test['Infarto silente']])


# Separate features and labels (for cross-validation later)
x = merged_df.drop(columns=['PatientId', 'DiagnosisName', 'Infarto silente']).copy()
y = merged_df['Infarto silente'].copy()


#%% RANDOM FOREST

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Random Forest classifier (8, 19)????
rf_classifier = RandomForestClassifier(n_estimators=18,criterion='gini')

# Train the model
rf_classifier.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Perform 5-fold cross-validation
rf = RandomForestClassifier(n_estimators=18,criterion='gini')
cv_scores = cross_val_score(rf, x, y, cv=5)

print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", cv_scores.mean())


#%% DECISION TREE

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Decision Tree classifier
dt_classifier = DecisionTreeClassifier(criterion='gini')

# Train the model
dt_classifier.fit(X_train_scaled, y_train)

# Make predictions
y_pred = dt_classifier.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Separate features and labels
x = merged_df.drop(columns=['PatientId', 'DiagnosisName', 'Infarto silente']).copy()
y = merged_df['Infarto silente'].copy()

# Perform 5-fold cross-validation
rf = DecisionTreeClassifier(criterion='gini')
cv_scores = cross_val_score(rf, x, y, cv=5)

print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", cv_scores.mean())



