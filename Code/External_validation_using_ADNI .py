#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
# Import pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# Import classes for data preprocessing
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Import machine learning algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsOneClassifier


# Import validation metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import RocCurveDisplay

# Import pyplot for data visualization
import matplotlib.pyplot as plt

# Set display options for pandas
pd.set_option('display.max_columns', None)

random.seed(42)
np.random.seed(42)



# Import pickle package to save sklearn ML trained models 
import pickle



# In[ ]:


# Load the ADNI data from a CSV file
ADNI_data = pd.read_csv('Path to data\\ADNI_data.csv')

# Select specific features of interest from the ADNI dataset
ADNI_features = ['RID','VISCODE','CDMEMORY', 'CDORIENT','CDJUDGE','CDCOMMUN','CDHOME','CDCARE','GDMEMORY','GDTOTAL','NPIC','NPIE','NPIG','NPII','NPIJ','FAQFINAN','FAQFORM','FAQSHOP','FAQGAME','FAQBEVG','FAQMEAL','FAQEVENT','FAQTV','FAQTRAVL','DX']   

# Create a DataFrame with the selected features
ADNI_selctedFeatures = ADNI_data[ADNI_features]

# Define mappings for FAQ and NPIQ features
FAQvalue_mapping = {'Normal (0)': 0,'Never did, but could do now (0)':0,'Never did, would have difficulty now (1)': 1,'Has difficulty, but does by self (1)': 1, 'Requires assistance (2)': 2, 'Dependent (3)': 3}
FAQfeatureToConvert = ['FAQFINAN','FAQFORM','FAQSHOP','FAQGAME','FAQBEVG','FAQMEAL','FAQEVENT','FAQTV','FAQTRAVL']
ADNI_selctedFeatures[FAQfeatureToConvert] = ADNI_selctedFeatures[FAQfeatureToConvert].replace(FAQvalue_mapping)

ADNI_selctedFeatures['GDMEMORY'] = ADNI_selctedFeatures['GDMEMORY'].replace({'No(0)': 0,'Yes(1)': 1})

NPIQ_feautres = ['NPIC','NPIE','NPIG','NPII','NPIJ']
NPIQvalue_mapping = {'No': 0,'Yes': 1}

ADNI_selctedFeatures[NPIQ_feautres] = ADNI_selctedFeatures[NPIQ_feautres].replace(NPIQvalue_mapping)

# Filter the DataFrame to select data first visits per subject
ADNI_selected_visits = ADNI_selctedFeatures[ADNI_selctedFeatures['VISCODE'] == 'bl'].groupby('RID').first().reset_index()

ADNI_selected_visits.groupby('DX').size() 

# Function to add fourth-visit diagnosis label
def ADNI_first_visit_plus_label_from_fourth_visit(df, df_first):
    visit_counts = df.groupby('RID')['VISCODE'].count()
    patients_with_4_or_more_visits = visit_counts[visit_counts >= 4].index
    fourth_visits = df[(df['RID'].isin(patients_with_4_or_more_visits))& (df['VISCODE'] == 'm36')]

    FourthState = fourth_visits[['RID','DX']]
        # rename the NACCUDSD column to NACCUDSD_4
    FourthState.rename(columns={'DX': 'DX4th'}, inplace=True)

    AD_after4years = pd.merge(df_first, FourthState, how='left', left_on=['RID'], right_on=['RID'])
    return AD_after4years

# Apply the function to add fourth-visit diagnosis labels
ADNI_subjects = ADNI_first_visit_plus_label_from_fourth_visit(ADNI_selctedFeatures ,ADNI_selected_visits)

# Replace diagnosis categories with numerical values
ADNI_subjects[['DX','DX4th']] = ADNI_subjects[['DX','DX4th']].replace({'CN': 0,'MCI': 1, 'Dementia':2})

# Rename columns based on the columns names from NACC dataset
columnsToRename={'CDMEMORY':'MEMORY', 'CDORIENT':'ORIENT', 'CDJUDGE':'JUDGMENT', 'CDCOMMUN':'COMMUN', 'CDHOME':'HOMEHOBB' ,'CDCARE':'PERSCARE',
        'GDMEMORY':'MEMPROB', 'GDTOTAL':'NACCGDS', 'NPIC':'AGIT', 'NPIE':'ANX', 'NPIG':'APA', 'NPII':'IRR', 'NPIJ':'MOT',
       'FAQFINAN':'BILLS', 'FAQFORM':'TAXES', 'FAQSHOP':'SHOPPING', 'FAQGAME':'GAMES', 'FAQBEVG':'STOVE', 'FAQMEAL':'MEALPREP', 'FAQEVENT':'EVENTS',
       'FAQTV':'PAYATTN', 'FAQTRAVL':'TRAVEL'}

ADNI_subjects = ADNI_subjects.rename(columns=columnsToRename)

# Split data into four diagnosis groups: CN vs AD, CN vs MCI, MCI vs AD, CN vs MCI vs AD
def ADNIsplit_data_by_diagnosis(df):
    """
    This function splits the input DataFrame into four different diagnosis groups: CN vs AD, 
    CN vs MCI, MCI vs AD, and CN vs MCI vs AD, and returns the feature matrix (X) and the 
    target variable (y) for each group.
    
    Parameters:
    -----------
    df: pandas DataFrame
        Input DataFrame containing the visits of patients to be split by diagnosis.
        
    Returns:
    --------
    tuple of pandas DataFrames and numpy arrays
        Four tuples, each containing the feature matrix (X) and the target variable (y)
        for each diagnosis group (CN vs AD, CN vs MCI, MCI vs AD, CN vs MCI vs AD).
    """
    # Select rows with only the first visit of each patient for CN vs AD group
    CNvsAD = df.loc[df['DX'].isin([0,2])]


    # Select rows with only the first visit of each patient for CN vs MCI group
    CNvsMCI = df.loc[df['DX'].isin([0,1])]


    # Select rows with only the first visit of each patient for MCI vs AD group
    MCIvsAD = df.loc[df['DX'].isin([1,2])]


    # Select rows with only the first visit of each patient for CN vs MCI vs AD group
    CNvsMCIvsAD = df.loc[df['DX'].isin([0,1,2])]


    return (CNvsAD), (CNvsMCI), (MCIvsAD), (CNvsMCIvsAD)



ADNI_CNvsAD, ADNI_CNvsMCI, ADNI_MCIvsAD, ADNI_CNvsMCIvsAD = ADNIsplit_data_by_diagnosis(ADNI_subjects)

# Function to split the data into feature matrix (X) and target variable (y)
def ADNI_X_y_Split(df, fourth=False):


    if fourth is True:
        X = df.drop(['RID','VISCODE','DX','DX4th'], axis=1)
        label_encoder = preprocessing.LabelEncoder()
        y = label_encoder.fit_transform(df['DX4th'])
  
    else:
        X = df.drop(['RID','VISCODE','DX','DX4th'], axis=1)
        label_encoder = preprocessing.LabelEncoder()
        y = label_encoder.fit_transform(df['DX'])
    
    # Return the feature array X and the encoded label array y.
    return X, y

def ADNI_4th_visit_split(df, task):
    from sklearn.model_selection import train_test_split
    
    testing_indices = df[df['DX4th'].notnull()].index
    #remining_indices = df.index.difference(testing_indices)

    test_df_ind = df.loc[testing_indices]
  
    
    if task == "CNvsAD":
        test_forth= test_df_ind.loc[test_df_ind['DX4th'].isin([0,2])]
    elif task == "CNvsMCI":
        test_forth= test_df_ind.loc[test_df_ind['DX4th'].isin([0,1])] 
    elif task == "MCIvsAD":
        test_forth= test_df_ind.loc[test_df_ind['DX4th'].isin([1,2])]
    elif task == "CNvsMCIvsAD":
        test_forth= test_df_ind.loc[test_df_ind['DX4th'].isin([0,1,2])] 

   
    
    return test_forth


#Extract data of CN vs AD subset classification and extract individuals with 4th visit for the prediction task
ADNI_CNvsAD_selectedFeautres = ADNI_CNvsAD.drop(['HOMEHOBB'], axis=1)

ADNI_CNvsAD_X, ADNI_CNvsAD_y = ADNI_X_y_Split(ADNI_CNvsAD_selectedFeautres)

ADNI_CNvsAD_4 = ADNI_4th_visit_split(ADNI_CNvsAD_selectedFeautres, 'CNvsAD')

ADNI_CNvsAD_4_X, ADNI_CNvsAD_4_y = ADNI_X_y_Split(ADNI_CNvsAD_4, fourth=True)

# Fill missing values with the mode for the feature data
ADNI_CNvsAD_X = ADNI_CNvsAD_X.fillna(ADNI_CNvsAD_X.mode().iloc[0])
ADNI_CNvsAD_4_X = ADNI_CNvsAD_4_X.fillna(ADNI_CNvsAD_4_X.mode().iloc[0])

# Load trained model on the NACC dataset
RF_CNvsAD_AfterFS_model = pickle.load(open('path to saved model\\RF_CNvsAD_AfterFS.sav', 'rb'))
SVM_CNvsAD_AfterFS_model = pickle.load(open('path to saved model\\SVM_CNvsAD_AfterFS.sav', 'rb'))

RF_CNvsAD_AfterFS_model_4th = pickle.load(open('path to saved model\\RF_CNvsAD_AfterFS_4th_Visit.sav', 'rb'))
SVM_CNvsAD_AfterFS_model_4th = pickle.load(open('path to saved model\\SVM_CNvsAD_AfterFS_4th_Visit.sav', 'rb'))

# Function to evaluate a model's performance
def ADNI_eval(model, X,y):
    predictions = model.predict(X)

        # Calculate the evaluation metrics
    acc = accuracy_score(y, predictions)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f1_sco = f1_score(y, predictions)

    # Print the evaluation metrics
    print('Test Performance')
    print('--------------------')
    print(confusion_matrix(y, predictions))
    print('Accuracy = ', (acc))
    print('Precision = ', (precision))
    print('Recall = ', (recall))
    print('F1_score = ', (f1_sco))
    

# Evaluate the trained machine learning models externally using ADNI data    
ADNI_eval(RF_CNvsAD_AfterFS_model, ADNI_CNvsAD_X, ADNI_CNvsAD_y)

ADNI_eval(SVM_CNvsAD_AfterFS_model, ADNI_CNvsAD_X, ADNI_CNvsAD_y)

ADNI_eval(RF_CNvsAD_AfterFS_model_4th, ADNI_CNvsAD_4_X, ADNI_CNvsAD_4_y)

ADNI_eval(SVM_CNvsAD_AfterFS_model_4th, ADNI_CNvsAD_4_X, ADNI_CNvsAD_4_y)

