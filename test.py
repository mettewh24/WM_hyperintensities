import utils
import numpy as np
import pandas as pd


# Test for list_performed_exams
def test_list_performed_exams():
    """
    Test the list_performed_exams function.

    Given Input:
    - patient_id: 'A', 'B', 'C'
    - dataframe: A sample DataFrame with patient exam data:
        {'PatientId': ['A', 'A', 'B', 'B', 'C'],
         'ExamName': ['X-ray', 'MRI', 'X-ray', 'CT', 'X-ray']}

    This function tests:
    - The function should return the correct list of performed exams for each patient.
    """
    # Create a sample DataFrame
    data = {'PatientId': ['A', 'A', 'B', 'B', 'C'],
            'ExamName': ['X-ray', 'MRI', 'X-ray', 'CT', 'X-ray']}
    df = pd.DataFrame(data)
    
    # Test the function
    assert np.array_equal(utils.list_performed_exams('A', df), np.array(['X-ray', 'MRI']))
    assert np.array_equal(utils.list_performed_exams('B', df), np.array(['X-ray', 'CT']))
    assert np.array_equal(utils.list_performed_exams('C', df), np.array(['X-ray']))


def test_patient_not_in_dataframe_list_performed_exams():
    """
    Test the list_performed_exams function for a patient not in the DataFrame.

    Given Input:
    - patient_id: 'D'
    - dataframe: A sample DataFrame with patient exam data:
        {'PatientId': ['A', 'A', 'B', 'B', 'C'],
         'ExamName': ['X-ray', 'MRI', 'X-ray', 'CT', 'X-ray']}

    This function tests:
    - The function should return an empty array when the specified patient ID is not found in the DataFrame.
    """
    # Create a sample DataFrame
    data = {'PatientId': ['A', 'A', 'B', 'B', 'C'],
            'ExamName': ['X-ray', 'MRI', 'X-ray', 'CT', 'X-ray']}
    df = pd.DataFrame(data)
    
    # Test the function with a patient ID that does not exist in the DataFrame
    assert np.array_equal(utils.list_performed_exams('D', df), np.array([]))


def test_empty_dataframe_list_of_performed_exams():
    """
    Test the list_performed_exams function with an empty DataFrame.

    Given Input:
    - patient_id: 'A'
    - dataframe: An empty DataFrame with columns ['PatientId', 'ExamName']

    This function tests:
    - The function should return an empty array when the DataFrame is empty.
    """
    # Create an empty DataFrame
    df = pd.DataFrame(columns=['PatientId', 'ExamName'])
    
    # Test the function with an empty DataFrame
    assert np.array_equal(utils.list_performed_exams('A', df), np.array([]))


# Test for num_of_repeated_exams
def test_num_of_repeated_exams():
    """
    Test the num_of_repeated_exams function.

    Given Input:
    - patient_id: 'A', 'B'
    - exam_name: 'X-ray', 'MRI', 'CT'
    - dataframe: A sample DataFrame with patient exam data:
        {'PatientId': ['A', 'A', 'B', 'B', 'C'],
         'ExamName': ['MRI', 'MRI', 'X-ray', 'CT', 'X-ray']}

    This function tests:
    - The function should return 0 when the specified exam is not found for the patient.
    - The function should return the correct count of repeated exams for the specified patient and exam.
    """
    # Create a sample DataFrame
    data = {'PatientId': ['A', 'A', 'B', 'B', 'C'],
            'ExamName': ['MRI', 'MRI', 'X-ray', 'CT', 'X-ray']}
    df = pd.DataFrame(data)
    
    # Test the function
    assert utils.num_of_repeated_exams('A', 'X-ray', df) == 0
    assert utils.num_of_repeated_exams('A', 'MRI', df) == 2
    assert utils.num_of_repeated_exams('B', 'CT', df) == 1


def test_patient_not_in_df_num_of_repeated_exams():
    """
    Test the num_of_repeated_exams function for a patient not in the DataFrame.

    Given Input:
    - patient_id: 'D'
    - exam_name: 'CT'
    - dataframe: A sample DataFrame with patient exam data:
        {'PatientId': ['A', 'A', 'B', 'B', 'C'],
         'ExamName': ['MRI', 'MRI', 'X-ray', 'CT', 'X-ray']}

    This function tests:
    - The function should return 0 when the specified patient ID is not found in the DataFrame.
    """
    # Create a sample DataFrame
    data = {'PatientId': ['A', 'A', 'B', 'B', 'C'],
            'ExamName': ['MRI', 'MRI', 'X-ray', 'CT', 'X-ray']}
    df = pd.DataFrame(data)
    
    # Test the function with a patient ID that does not exist in the DataFrame
    assert utils.num_of_repeated_exams('D', 'CT', df) == 0

def test_exam_not_done_num_of_repeated_exams():
    """
    Test the num_of_repeated_exams function for an exam not done by the patient.

    Given Input:
    - patient_id: 'A'
    - exam_name: 'CT'
    - dataframe: A sample DataFrame with patient exam data:
        {'PatientId': ['A', 'A', 'B', 'B', 'C'],
         'ExamName': ['MRI', 'MRI', 'X-ray', 'CT', 'X-ray']}

    This function tests:
    - The function should return 0 when the specified exam has not been done by the patient.
    """
    # Create a sample DataFrame
    data = {'PatientId': ['A', 'A', 'B', 'B', 'C'],
            'ExamName': ['MRI', 'MRI', 'X-ray', 'CT', 'X-ray']}
    df = pd.DataFrame(data)
    
    # Test the function with an exam that has not been done by the patient
    assert utils.num_of_repeated_exams('A', 'CT', df) == 0

def test_empty_dataframe_num_of_repeated_exams():
    """
    Test the num_of_repeated_exams function with an empty DataFrame.

    Given Input:
    - patient_id: 'A'
    - exam_name: 'MRI'
    - dataframe: An empty DataFrame with columns ['PatientId', 'ExamName']

    This function tests:
    - The function should return 0 when the DataFrame is empty.
    """    
    # Create an empty DataFrame
    df = pd.DataFrame(columns=['PatientId', 'ExamName'])
    
    # Test the function with an empty DataFrame
    assert utils.num_of_repeated_exams('A', 'MRI', df) == 0
