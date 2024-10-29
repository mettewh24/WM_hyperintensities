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

