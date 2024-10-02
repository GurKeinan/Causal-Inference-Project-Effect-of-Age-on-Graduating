"""
This module provides functions for preprocessing a dataset that contains 
information about individuals, specifically focusing on categorizing occupations 
and qualifications of parents, filtering records, and saving the processed data 
to a CSV file.

The main functionalities of the module include:

1. Categorizing occupation codes into descriptive categories.
2. Categorizing qualification codes into descriptive categories.
3. Categorizing previous qualification codes into descriptive categories.
4. Loading data, processing it by applying the above categorizations, 
   filtering based on specific criteria, and saving the cleaned data.

Usage:
    To use this module, simply import it and call the relevant functions 
    as needed. The processed data will be saved as 'data/processed_data.csv'.

Dependencies:
    - pandas: For data manipulation and analysis.
"""

import pandas as pd

# Function to categorize occupations
def categorize_occupation(occupation):
    """
    Categorize an occupation code into a descriptive occupation category.

    Parameters:
    occupation (int): The occupation code to categorize.

    Returns:
    str: A string representing the occupation category. Possible categories are:
        - 'Management'
        - 'Professionals'
        - 'Technicians'
        - 'Administrative'
        - 'Service and Sales'
        - 'Agriculture and Fishery'
        - 'Craft and Related Trades'
        - 'Plant and Machine Operators'
        - 'Elementary Occupations'
        - 'Armed Forces'
        - 'Other' (for any occupation code not listed above)
    """
    if occupation in [1, 2, 17, 18]:
        return 'Management'
    elif occupation in [3, 19, 20, 21, 22]:
        return 'Professionals'
    elif occupation in [4, 23, 24, 25, 26]:
        return 'Technicians'
    elif occupation in [5, 27, 28, 29]:
        return 'Administrative'
    elif occupation in [6, 30, 31, 32, 33]:
        return 'Service and Sales'
    elif occupation in [7, 34, 35]:
        return 'Agriculture and Fishery'
    elif occupation in [8, 36, 37, 38, 39]:
        return 'Craft and Related Trades'
    elif occupation in [9, 40, 41, 42]:
        return 'Plant and Machine Operators'
    elif occupation in [10, 43, 44, 45, 46]:
        return 'Elementary Occupations'
    elif occupation in [11, 14, 15, 16]:
        return 'Armed Forces'
    else:
        return 'Other'

# Function to categorize parent qualifications
def categorize_qualification(qualification):
    """
    Categorizes the given qualification code into a descriptive string.

    Parameters:
    qualification (int): The qualification code to be categorized.

    Returns:
    str: A string describing the category of the qualification. Possible categories are:
        - 'Complete Secondary Education'
        - 'Incomplete Secondary Education'
        - 'Higher Education - Undergraduate'
        - 'Higher Education - Postgraduate'
        - 'Basic Education (1st and 2nd Cycle)'
        - 'Basic Education (3rd Cycle)'
        - 'Technological Specialization'
        - 'No Formal Education'
        - 'Unknown'
        - 'Other Specific Qualifications'
    """
    if qualification in [1, 10, 11, 12, 13, 14, 15, 16]:
        return 'Complete Secondary Education'
    elif qualification in [7, 8, 17, 19, 20]:
        return 'Incomplete Secondary Education'
    elif qualification in [2, 3, 4, 5, 6, 30, 31, 32]:
        return 'Higher Education - Undergraduate'
    elif qualification in [33, 34]:
        return 'Higher Education - Postgraduate'
    elif qualification in [27, 28]:
        return 'Basic Education (1st and 2nd Cycle)'
    elif qualification in [9, 18, 21]:
        return 'Basic Education (3rd Cycle)'
    elif qualification == 29:
        return 'Technological Specialization'
    elif qualification in [25, 26]:
        return 'No Formal Education'
    elif qualification == 24:
        return 'Unknown'
    else:
        return 'Other Specific Qualifications'

# Function to categorize previous qualification
def categorize_previous_qualification(qualification):
    """
    Categorizes the previous qualification based on the given qualification code.

    Parameters:
    qualification (int): The code representing the previous qualification.

    Returns:
    str: A string describing the category of the previous qualification. Possible 
         categories are:
            - 'Complete Secondary Education'
            - 'Incomplete Secondary Education'
            - 'Higher Education - Bachelor/Degree'
            - 'Higher Education - Master'
            - 'Higher Education - Doctorate'
            - 'Frequency of Higher Education'
            - 'Basic Education (3rd Cycle)'
            - 'Basic Education (2nd Cycle)'
            - 'Technological Specialization'
            - 'Professional Higher Technical'
            - 'Other'
    """
    if qualification == 1:
        return 'Complete Secondary Education'
    elif qualification in [7, 8, 9, 10, 11]:
        return 'Incomplete Secondary Education'
    elif qualification in [2, 3, 15]:
        return 'Higher Education - Bachelor/Degree'
    elif qualification in [4, 17]:
        return 'Higher Education - Master'
    elif qualification == 5:
        return 'Higher Education - Doctorate'
    elif qualification == 6:
        return 'Frequency of Higher Education'
    elif qualification == 12:
        return 'Basic Education (3rd Cycle)'
    elif qualification == 13:
        return 'Basic Education (2nd Cycle)'
    elif qualification == 14:
        return 'Technological Specialization'
    elif qualification == 16:
        return 'Professional Higher Technical'
    else:
        return 'Other'

FATHER_OCCUPATION_STRING = 'Father\'s occupation'
MOTHER_OCCUPATION_STRING = 'Mother\'s occupation'
FATHER_QUALIFICATION_STRING = 'Father\'s qualification'
MOTHER_QUALIFICATION_STRING = 'Mother\'s qualification'
PREVIOUS_QUALIFICATION_STRING = 'Previous qualification'
RAW_DATA_FILE = 'data/raw_data.csv'
PROCESSED_DATA_FILE = 'data/processed_data.csv'


# Load the data
df = pd.read_csv(RAW_DATA_FILE)

# Create a binary column indicating if the individual is an adult (age >= 21)
df['Adult'] = (df['Age at enrollment'] >= 21).astype(int)
df = df.drop(columns=['Age at enrollment'])

# 1. Replace occupation and qualification categories
df[FATHER_OCCUPATION_STRING] = df[FATHER_OCCUPATION_STRING].apply(categorize_occupation)
df[MOTHER_OCCUPATION_STRING] = df[MOTHER_OCCUPATION_STRING].apply(categorize_occupation)
df[FATHER_QUALIFICATION_STRING] = df[FATHER_QUALIFICATION_STRING].apply(categorize_qualification)
df[MOTHER_QUALIFICATION_STRING] = df[MOTHER_QUALIFICATION_STRING].apply(categorize_qualification)
df[PREVIOUS_QUALIFICATION_STRING] = df[PREVIOUS_QUALIFICATION_STRING].apply(categorize_previous_qualification)

# 2. Omit records not in top 5 common qualifications for parents
top_5_father_qual = df[FATHER_QUALIFICATION_STRING].value_counts().nlargest(5).index
top_5_mother_qual = df[MOTHER_QUALIFICATION_STRING].value_counts().nlargest(5).index

df = df[df[FATHER_QUALIFICATION_STRING].isin(top_5_father_qual) &
         df[MOTHER_QUALIFICATION_STRING].isin(top_5_mother_qual)]

# Filter for records with nationality as Portuguese, then drop nationality and international columns
df = df[df['Nacionality'] == 1]
df = df.drop(columns=['Nacionality', 'International'])

# List and remove all columns starting with 'Curricular'
curricular_cols = [col for col in df.columns if col.startswith('Curricular')]
df = df.drop(columns=curricular_cols)

# Remove the columns 'Debtor' and 'Tuition fees up to date'
df = df.drop(columns=['Debtor', 'Tuition fees up to date'])

# 3. Omit records not in top 4 common previous qualifications
top_4_prev_qual = df[PREVIOUS_QUALIFICATION_STRING].value_counts().nlargest(4).index
df = df[df[PREVIOUS_QUALIFICATION_STRING].isin(top_4_prev_qual)]

# Save the processed data to a new CSV file
df.to_csv('data/processed_data.csv', index=False)

print("Preprocessing complete. Data saved to " + PROCESSED_DATA_FILE)
print(f"Original data shape: {pd.read_csv(RAW_DATA_FILE).shape}")
print(f"Preprocessed data shape: {df.shape}")
