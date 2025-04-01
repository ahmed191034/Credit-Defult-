# Credit-Defult-
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import warnings
import os


# Load the dataset
a1 = pd.read_csv(r"D:\Machine Learning Project\Machine Learning Project\Data Set\case_study1.csv")
a2 = pd.read_csv(r"D:\Machine Learning Project\Machine Learning Project\Data Set\case_study2.csv")


df1 = a1.copy()
df2 = a2.copy()



# Remove nulls
df1 = df1.loc[df1['Age_Oldest_TL'] != -99999]


''''The values which were nulss were automaticaly assigned as -99999 '''

columns_to_be_removed = []

for i in df2.columns:
    if df2.loc[df2[i] == -99999].shape[0] > 10000:
        columns_to_be_removed .append(i)



df2 = df2.drop(columns_to_be_removed, axis =1)



for i in df2.columns:
    df2 = df2.loc[ df2[i] != -99999 ]



# Merge the two dataframes, inner join so that no nulls are present
df = pd. merge ( df1, df2, how ='inner', left_on = ['PROSPECTID'], right_on = ['PROSPECTID'] )

#Feature Selection


# checking how many columns are categorical
for i in df.columns:
    if df[i].dtype == 'object':
        print(i)
   
'''
We want to selct those features which are helping us predicting the Target variable[Aproved Flags]
 since our Traget Variable is categorical and we have other Categorical varibles what we will do is
 perform a Staistical test call Chi squre becuse Categorical vs Categorical


Null Hypothesis (H₀):
There is no association between the categorical predictor variable 
(e.g., MARITALSTATUS, EDUCATION, etc.) and the target variable (Approved_Flag).



The explanation of using the Chi-Square test to determine whether there is a 
significant relationship between categorical variables is also correct.

If p-value > 0.05, we fail to reject the null hypothesis:
    
(meaning there is no significant association).

If p-value < 0.05, we reject the null hypothesis:
    
(meaning there is a significant association).

'''

# Chi-square test
for i in ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']:
    chi2, pval, _, _ = chi2_contingency(pd.crosstab(df[i], df['Approved_Flag']))
    print(i, '---', pval)
'''
The results indicate that all predictor variables have p-values much smaller than 0.05, 
leading us to reject the null hypothesis and conclude that these variables are statistically
associated with the target variable (Approved_Flag).
'''


# VIF for numerical columns
numeric_columns = []
for i in df.columns:
    if df[i].dtype != 'object' and i not in ['PROSPECTID','Approved_Flag']:
        numeric_columns.append(i)

#Multicolinearity Check

# VIF sequentially check

vif_data = df[numeric_columns]
total_columns = vif_data.shape[1]
columns_to_be_kept = []
column_index = 0

for i in range (0,total_columns):
    
    vif_value = variance_inflation_factor(vif_data, column_index)
    print (column_index,'---',vif_value)
    
    
    if vif_value <= 6:
        columns_to_be_kept.append( numeric_columns[i] )
        column_index = column_index+1
    
    else:
        vif_data = vif_data.drop([ numeric_columns[i] ] , axis=1)

'''
There were 72 columns orignally and during our vif check of Multicolinearity 
33 columns are removed means they were predictable.

 '''  
 
# Check ANOVA for numerical columns
from scipy.stats import f_oneway

columns_to_be_kept_numerical = []

for i in columns_to_be_kept:
    a = list(df[i])  
    b = list(df['Approved_Flag'])  
    
    group_P1 = [value for value, group in zip(a, b) if group == 'P1']
    group_P2 = [value for value, group in zip(a, b) if group == 'P2']
    group_P3 = [value for value, group in zip(a, b) if group == 'P3']
    group_P4 = [value for value, group in zip(a, b) if group == 'P4']

    f_statistic, p_value = f_oneway(group_P1, group_P2, group_P3, group_P4)

    if p_value <= 0.05:
        columns_to_be_kept_numerical.append(i)
        


# Final column selection
features = columns_to_be_kept_numerical + ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
df = df[features + ['Approved_Flag']]
        
df['MARITALSTATUS'].unique()    
df['EDUCATION'].unique()
df['GENDER'].unique()
df['last_prod_enq2'].unique()
df['first_prod_enq2'].unique()
         
 
# Feature Enginnering
'''
Label Encoding  
Since I have 5 categorical columns i need to update them to certain vlaues
Education is the only column having ordinal encoding the rest of them is
label encoding which is OHE                             

'''

# Ordinal Encoding

df.loc[df['EDUCATION'] == 'SSC', 'EDUCATION'] = 1
df.loc[df['EDUCATION'] == '12TH', 'EDUCATION'] = 2
df.loc[df['EDUCATION'] == 'GRADUATE', 'EDUCATION'] = 2
df.loc[df['EDUCATION'] == 'UNDER GRADUATE', 'EDUCATION'] = 3
df.loc[df['EDUCATION'] == 'POST-GRADUATE', 'EDUCATION'] = 4
df.loc[df['EDUCATION'] == 'OTHERS', 'EDUCATION'] = 1
df.loc[df['EDUCATION'] == 'PROFESSIONAL', 'EDUCATION'] = 2

df['EDUCATION'] = df['EDUCATION'].astype(int)

df['EDUCATION'].value_counts()

# Label encoding


df_encoded = pd.get_dummies(df, columns=['MARITALSTATUS','GENDER', 'last_prod_enq2' ,'first_prod_enq2'])
df_encoded.info()

df_encoded = pd.get_dummies(df, columns=['MARITALSTATUS', 'GENDER', 'last_prod_enq2' ,'first_prod_enq2'], dtype=int)


#Data spliting

y = df_encoded['Approved_Flag']
x = df_encoded. drop ( ['Approved_Flag'], axis = 1 )


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Machine Learing model fitting


# 1. Random Forest
rf_classifier = RandomForestClassifier(n_estimators = 200, random_state=42)
rf_classifier.fit(x_train, y_train)
y_pred = rf_classifier.predict(x_test)

print("Random Forest")
accuracy = accuracy_score(y_test, y_pred)
print ()
print(f'Accuracy: {accuracy}')
print ()
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    print()


# 2. xgboost

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

xgb_classifier = xgb.XGBClassifier(objective='multi:softmax',  num_class=4)



y = df_encoded['Approved_Flag']
x = df_encoded. drop ( ['Approved_Flag'], axis = 1 )


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)


xgb_classifier.fit(x_train, y_train)
y_pred = xgb_classifier.predict(x_test)

print("X-g boost")

accuracy = accuracy_score(y_test, y_pred)
print ()
print(f'Accuracy: {accuracy:.2f}')
print ()

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    print()

# Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

# Prepare target and features
y = df_encoded['Approved_Flag']
x = df_encoded.drop(['Approved_Flag'], axis=1)

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

# Create a Logistic Regression classifier for multiclass classification
lr_classifier = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

# Fit the classifier to the training data
lr_classifier.fit(x_train, y_train)

# Make predictions on the test set
y_pred = lr_classifier.predict(x_test)



print("Logistic Regression")
# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print()

precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)
# Assuming there are four classes, printing metrics for each class
for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]:.2f}")
    print(f"Recall: {recall[i]:.2f}")
    print(f"F1 Score: {f1[i]:.2f}")
    print()


