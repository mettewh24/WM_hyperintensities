#%% Libraries import

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind

#%% Search for correlations

# Load the data in a pandas dataframe
dati_esami_al_31122020=pd.read_csv('./data/dati_esami_al_31122020.csv')

#Filter the dataframe to remove unnecessary columns (excluded Note, GroupName,GroupId,DiagnosisId)
filtered_dati_esami_al_31122020=dati_esami_al_31122020[["DiagnosisName","ExamName","Outcome","PatientId","Date","ExamId"]]

#Remove rows with missing Outcome value (clearly not useful for the analysis)
filtered_dati_esami_al_31122020=filtered_dati_esami_al_31122020.dropna(subset=["Outcome"], inplace=False, ignore_index=True)

#Change decimal separator from comma to dot, to avoid errors when using pd.to_numeric
filtered_dati_esami_al_31122020["Outcome"]=filtered_dati_esami_al_31122020["Outcome"].str.replace(',', '.') #replace comma with dot 

#Convert to numeric the Outcome column 
#NOTE: errors='coerce' is used to force conversion of non-numeric values to NaN
filtered_dati_esami_al_31122020["Outcome"]=pd.to_numeric(filtered_dati_esami_al_31122020["Outcome"], errors='coerce')

#For each patient, for each diagnosis, calculate the mean of the outcomes
pivot_df = filtered_dati_esami_al_31122020.pivot_table(index=['DiagnosisName', 'PatientId'], columns='ExamName', values='Outcome', aggfunc='mean')
pivot_df = pivot_df.reset_index(drop=False)


#%% Perform t-tests

#NOTE: Check Hb exam, not clear why most have 0 value
#TODO: Check for warnings of ttest_ind 

# Define the groups to compare
group1 = "HbS/HbC"
group2 = "HbS/HbS"
group3 = "HbS/βthal°"


#TODO: Tests
def t_test(group1_name, group2_name, dataframe):
    #initialize dictionary to store t-test results
    t_test = {}

    # Group the dataframe by DiagnosisName
    diagnosis_groups = dataframe.groupby("DiagnosisName")

    # Loop through each column and perform t-tests
    for column in dataframe.columns:
        if column not in ['DiagnosisName', 'PatientId']:
            group1 = diagnosis_groups.get_group(group1_name)[column].dropna(ignore_index=True)
            group2 = diagnosis_groups.get_group(group2_name)[column].dropna(ignore_index=True)
 
            # Perform t-test between group1 and group2
            t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
            t_test[column] = {'t_stat': t_stat, 'p_value': p_value}
    
    #Convert to dataframe and drop rows with missing values
    t_test_df = pd.DataFrame(t_test).T
    t_test_df.dropna(inplace=True)


    return t_test_df


t_test_1_2_df=t_test(group1,group2,pivot_df)
t_test_1_3_df=t_test(group1,group3,pivot_df)
t_test_2_3_df=t_test(group2,group3,pivot_df)

# Filter the results to keep only significant p-values
t_test_1_3_df=t_test_1_3_df[t_test_1_3_df["p_value"]<0.05]
t_test_1_3_df.reset_index(inplace=True)

t_test_2_3_df=t_test_2_3_df[t_test_2_3_df["p_value"]<0.05]
t_test_2_3_df.reset_index(inplace=True)

t_test_1_2_df=t_test_1_2_df[t_test_1_2_df["p_value"]<0.05]
t_test_1_2_df.reset_index(inplace=True)

#%% Save the results to a text file

lines=[]
lines.append('T-test results for HbS/HbC vs HbS/βthal°\n')
lines.append(t_test_1_3_df.to_string(index=False)+'\n\n')

lines.append('T-test results for HbS/HbS vs HbS/βthal°\n')
lines.append(t_test_2_3_df.to_string(index=False)+'\n\n')

lines.append('T-test results for HbS/HbC vs HbS/HbS\n')
lines.append(t_test_1_2_df.to_string(index=False)+'\n\n')

with open(f"./summary/t-tests.txt", 'w') as file:
        file.writelines(lines)

file.close()


#%% Plot some of the outcomes, gruoped by Diagnosis
df=pivot_df[["DiagnosisName","Hb S","GB","MCV",'Hb C']]

palette = {
    "HbS/HbS": "green",
    "HbS/HbC": "blue",
    "HbS/βthal°": "red"
}

g=sns.PairGrid(df, height=2, hue="DiagnosisName",palette=palette)
g.map_diag(sns.histplot,alpha=0.5)
g.map_offdiag(sns.scatterplot,alpha=0.5,s=10)

g.add_legend(title="Diagnosis")

# %%
