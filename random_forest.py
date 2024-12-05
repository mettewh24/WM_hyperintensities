#%%
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut,StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, matthews_corrcoef
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

# Separate features and labels (for cross-validation later)
x = merged_df.drop(columns=['PatientId', 'DiagnosisName', 'Infarto silente']).copy()
y = merged_df['Infarto silente'].copy()

X_train, X_test, y_train, y_test = train_test_split(x, y,stratify=y, test_size=0.3)

#%% RANDOM FOREST

# Define the pipeline steps for the Random Forest classifier (with standardization)
rf_classifier = Pipeline([
    ('scaler', StandardScaler()),  # Standardize the features
    ('classifier', RandomForestClassifier(n_estimators=18,criterion='gini'))  # Random Forest classifier    
])

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
print(f"Balanced accuracy: {balanced_accuracy:.2f}")

mcc = matthews_corrcoef(y_test, y_pred)
print(f"Matthews Correlation Coefficient: {mcc:.2f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Perform 5-fold cross-validation
cv_scores = cross_val_score(rf_classifier, x, y, cv=5)

print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", cv_scores.mean())


#%% DECISION TREE

# Define the pipeline steps for the decision tree classifier (with standardization)
dt_classifier = Pipeline([
    ('scaler', StandardScaler()), # Standardize the features
    ('classifier', DecisionTreeClassifier(criterion='gini')) # Decision Tree classifier
])

# Train the pipeline
dt_classifier.fit(X_train, y_train)

# Make predictions
y_pred = dt_classifier.predict(X_test)

# Evaluate the model
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
print(f"Balanced accuracy: {balanced_accuracy:.2f}")

mcc = matthews_corrcoef(y_test, y_pred)
print(f"Matthews Correlation Coefficient: {mcc:.2f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Separate features and labels
x = merged_df.drop(columns=['PatientId', 'DiagnosisName', 'Infarto silente']).copy()
y = merged_df['Infarto silente'].copy()

# Perform 5-fold cross-validation
cv_scores = cross_val_score(dt_classifier, x, y, cv=5)

print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", cv_scores.mean())



#%% DECISION TREE WITH LEAVE-ONE-OUT CROSS-VALIDATION

# Define the pipeline steps
dt_classifier = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier',DecisionTreeClassifier(criterion='gini',random_state=42))
])

# Initialize Leave-One-Out Cross-Validation
loo = LeaveOneOut()

# Perform LOOCV
y_true, y_pred = [], []
for train_index, test_index in loo.split(x):
    X_train_LOOCV, X_test_LOOCV = x.iloc[train_index], x.iloc[test_index]
    y_train_LOOCV, y_test_LOOCV = y.iloc[train_index], y.iloc[test_index]
    
    # Train the pipeline
    dt_classifier.fit(X_train_LOOCV, y_train_LOOCV)
    
    # Make predictions
    y_pred.append(dt_classifier.predict(X_test_LOOCV)[0])
    y_true.append(y_test_LOOCV.values[0])

# Evaluate the model
balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
print(f"Balanced accuracy: {balanced_accuracy:.2f}")

mcc = matthews_corrcoef(y_true, y_pred)
print(f"Matthews Correlation Coefficient: {mcc:.2f}")

print("Classification Report:")
print(classification_report(y_true, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))



#%% Let's try to find dependece of performance with the split of the dataset

# Separate features and labels
x = merged_df.drop(columns=['PatientId', 'DiagnosisName', 'Infarto silente']).copy()
y = merged_df['Infarto silente'].copy()

# Define the pipeline steps
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize the features
    ('classifier', DecisionTreeClassifier(criterion='gini',random_state=42))  # Decision Tree classifier
])

# Number of repetitions
n_repeats = 100

# Store the cross-validation scores
all_cv_scores = []

cv = StratifiedKFold(n_splits=5, shuffle=True)

for i in range(n_repeats):
    cv_scores = cross_val_score(pipeline, x, y, cv=cv, scoring='balanced_accuracy')
    all_cv_scores.append(cv_scores)
    print(f"Iteration {i+1}: Cross-validation scores: {cv_scores}")

# Convert to a numpy array for easier analysis
all_cv_scores = np.array(all_cv_scores)

# Calculate the overall mean and standard deviation
overall_mean_cv_score = np.mean(all_cv_scores)
overall_std_cv_score = np.std(all_cv_scores)

print(f"Overall mean cross-validation score: {overall_mean_cv_score:.2f}")
print(f"Overall standard deviation of cross-validation scores: {overall_std_cv_score:.2f}")