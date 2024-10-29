#%% Importing libraries

import numpy as np 
import pandas as pd
import seaborn as sns
from utils import list_performed_exams, num_of_repeated_exams

# %% Print a summary of the data in the dataframe to a text file

def print_patient_summary(dataframe: pd.DataFrame, filename: str):
    """
    Print a summary of the data in the dataframe to a text file.

    The summary includes the total number of patients and a detailed list of performed exams for each patient,
    along with the count of how many times each exam was repeated.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame containing patient exam data. It should have at least the following columns:
                              - 'PatientId': A unique identifier for each patient.
                              - 'ExamName': The name of the exam performed.
    filename (str): The name of the file to write the summary to. The file will be saved in the './summary/' directory.

    Returns:
    None
    """
    summary_lines = []
    unique_patient_ids = dataframe['PatientId'].unique()
    
    summary_lines.append(f"Total number of patients in dataframe: {len(unique_patient_ids)}\n\n")
    
    for patient_id in unique_patient_ids:
        summary_lines.append(f"Patient ID: {patient_id}\n")
        performed_exams = list_performed_exams(patient_id, dataframe)
        times_exam_repeated = {exam: num_of_repeated_exams(patient_id, exam, dataframe) for exam in performed_exams}
        
        summary_lines.append("\tPerformed exams: \n")
        for exam, count in times_exam_repeated.items():
            summary_lines.append(f"\t{exam}: {count} time(s)\n")
        summary_lines.append("\n\n")
    with open(f"./summary/patient_summary_{filename}.txt", 'w') as file:
        file.writelines(summary_lines)

def print_exam_summary(dataframe: pd.DataFrame, filename: str):
    """
    Print a summary of the exams in the dataframe to a text file.

    The summary includes the total number of unique exams and details for each exam,
    such as the number of unique patients who performed the exam and the total number of times the exam was performed.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame containing patient exam data. It should have at least the following columns:
                              - 'ExamName': The name of the exam performed.
                              - 'PatientId': A unique identifier for each patient.
    filename (str): The name of the file to write the summary to. The file will be saved in the './summary/' directory.

    Returns:
    None
    """
    summary_lines = []
    unique_exam = dataframe['ExamName'].unique()

    summary_lines.append(f"Total number of exams in dataframe: {len(unique_exam)}\n\n")
    for exam in unique_exam:
        summary_lines.append(f"Exam Name: {exam}\n")

        temp = dataframe[dataframe['ExamName'] == exam]
        summary_lines.append(f"Performed by {temp['PatientId'].nunique()} patients for a total of {temp.shape[0]} times\n\n")
    with open(f"./summary/exam_summary_{filename}.txt", 'w') as file:
        file.writelines(summary_lines)


#%%  Charcterization of dati_clinici_ematologici

# Load the data
dati_clinici_ematologici=pd.read_csv('./data/dati_clinici_ematologici.csv')

# Print summary for the dati_clinici_ematologici dataframe
print_patient_summary(dati_clinici_ematologici, "dati_clinici_ematologici")

print_exam_summary(dati_clinici_ematologici, "dati_clinici_ematologici")


#%% Charcterization of dati_esami_al_31122020

#Load the data
dati_esami_al_31122020=pd.read_csv('./data/dati_esami_al_31122020.csv')

#Print summary for the dati_esami_al_31122020 dataframe
print_patient_summary(dati_esami_al_31122020,"dati_esami_al_31122020")

print_exam_summary(dati_esami_al_31122020, "dati_esami_al_31122020")

# %% Data pre-processing of dati_parametri_al_31122020

# Load the data
dati_parametri_al_31122020=pd.read_csv('./data/dati_parametri_al_31122020.csv')
sns.jointplot(data=dati_parametri_al_31122020, x='BloodPressureMax', y='BloodPressureMin', kind='scatter')

#Remove Outliers with no physical/medical meaning

#Weight(Kg)
dati_parametri_al_31122020.loc[dati_parametri_al_31122020['Weight(Kg)']>200,'Weight(Kg)']=np.nan
dati_parametri_al_31122020.loc[dati_parametri_al_31122020['Weight(Kg)']<5,'Weight(Kg)']=np.nan

#Height(cm)
dati_parametri_al_31122020.loc[dati_parametri_al_31122020['Height(cm)']>250,'Height(cm)']=np.nan
dati_parametri_al_31122020.loc[dati_parametri_al_31122020['Height(cm)']<30,'Height(cm)']=np.nan
sns.jointplot(data=dati_parametri_al_31122020, x='Weight(Kg)', y='Height(cm)', kind='scatter')

#Correction on inverted values for max and min blood pressure
condition = dati_parametri_al_31122020['BloodPressureMax'] < dati_parametri_al_31122020['BloodPressureMin']
dati_parametri_al_31122020.loc[condition, ['BloodPressureMax', 'BloodPressureMin']] = dati_parametri_al_31122020.loc[condition, ['BloodPressureMin', 'BloodPressureMax']].values

#BloodPressureMax <20 removal (impossible to have max blood pressure lower than 20)
dati_parametri_al_31122020.loc[dati_parametri_al_31122020['BloodPressureMax']<20,'BloodPressureMax']=np.nan

#BloodPressureMin <20 removal (impossible to have min blood pressure lower than 20)
dati_parametri_al_31122020.loc[dati_parametri_al_31122020['BloodPressureMin']<20,'BloodPressureMin']=np.nan
sns.jointplot(data=dati_parametri_al_31122020, x='BloodPressureMax', y='BloodPressureMin', kind='scatter')

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
