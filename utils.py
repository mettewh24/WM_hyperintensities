import numpy as np
import pandas as pd


def list_performed_exams(patient_id:str,dataframe:pd.DataFrame)->np.ndarray:
    """
    List all performed exams for a given PatientId in the DataFrame.

    Parameters:
    patient_id (str): The ID of the patient.
    dataframe (pd.DataFrame): The DataFrame containing patient exam data.

    Returns:
    np.ndarray: An array of unique exam names available for the specified patient.
    """
    
    return dataframe[dataframe['PatientId'] == patient_id]['ExamName'].unique()

#NOTE: function DOES NOT raise a warning if patient_id is not in the dataframe, returns 0 
def num_of_repeated_exams(patient_id:str, exam_name:str, dataframe:pd.DataFrame) -> int:
    """
    Count the number of times a specific exam was repeated for a given PatientId.

    Parameters:
    patient_id (str): The ID of the patient.
    exam_name (str): The name of the exam.
    dataframe (pd.DataFrame): The DataFrame containing patient exam data.

    Returns:
    int: Number of times the specified exam was repeated for the specified patient.
    """
    # Exclude the rows that do not match patient_id and exam_name
    dataframe_filtered = dataframe[(dataframe['PatientId'] == patient_id) & (dataframe['ExamName'] == exam_name)]
    return dataframe_filtered.shape[0]
