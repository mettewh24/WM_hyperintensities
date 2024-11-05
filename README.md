# WM_hyperintensities
Study of correlation between ematological data and white matter hyperintesities

## File Descriptions

### utils.py
Contains utility functions for processing patient exam data:
- `list_performed_exams(patient_id: str, dataframe: pd.DataFrame) -> np.ndarray`: Lists all performed exams for a given patient.
- `num_of_repeated_exams(patient_id: str, exam_name: str, dataframe: pd.DataFrame) -> int`: Counts the number of times a specific exam was repeated for a given patient.
- `is_exam_available(patient_id: str, exam_name: str, dataframe: pd.DataFrame) -> bool`: Checks if a specific exam is available for a given patient.
- `num_of_patients_per_exam(dataframe: pd.DataFrame, exam_name: str) -> int`: Counts the number of patients who have undergone a specific exam.

### test.py
Contains unit tests for the functions in `utils.py`:
- Tests for `list_performed_exams`
- Tests for `num_of_repeated_exams`
- Tests for `is_exam_available`
- Tests for `num_of_patients_per_exam`

### preliminary_analysis.py
Performs preliminary analysis on clinical and exam data:
- Loads and preprocesses data from CSV files.
- Prints summaries of patient and exam data to text files.
- Removes outliers and corrects data inconsistencies.

### correlation.py
Searches for correlations in exam data and performs statistical tests:
- Loads and preprocesses exam data.
- Performs t-tests to compare outcomes between different diagnosis groups.
- Saves t-test results to a text file.
- Plots outcomes grouped by diagnosis.

### summary folder
Contains the `.txt` files with the results of the patient and exams summary:
- `exam_summary_dati_clinici_ematologici.txt`: summary of the exams in `dati_clinici_ematologici.csv`. For each exam, the number of patient and the number of times that the exam was performed is reported, as well as the total number of exams in the dataframe
- `patient_summary_dati_clinici_ematologici.txt`: summary of the patients in `dati_clinici_ematologici.csv`. For each patient, the list of exams performed and the number of times each exams was repeated is reported, as well as the total number of patients in the dataframe.
- `exam_summary_dati_esami_al_31122020.txt`:summary of the exams in `dati_esami_al_31122020.csv`. For each exam, the number of patient and the number of times that the exam was performed is reported, as well as the total number of exams in the dataframe
- `patient_summary_dati_esami_al_31122020.txt`: summary of the patients in `dati_esami_al_31122020.csv`. For each patient, the list of exams performed and the number of times each exams was repeated is reported, as well as the total number of patients in the dataframe.
- `t-tests.txt`: contains the results of the t-test for the exams that have `p<0.05`, for all the possible diagnosis group pairs.

### Plots folder 
Folder containing all the saved plots:
- `blood_pressure.png`: jointplot of min and max blood pressure.
- `blood_pressure_clean.png`: jointplot of min and max blood pressure, after removing outliers.
- `height_weight.png`: jointplot of weight and height.
- `height_weight_clean.png`: jointplot of weight and height, after removing outliers.
- `pairplot.png`: pairplot of some of the variables with `p<0.05` from the t-test. 