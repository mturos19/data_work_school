# Mark Turos
# 199015662
# CO3095



## Part 1 - Building a basic predictive model

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)#Ignore FutureWarningPopup
warnings.simplefilter(action='ignore', category=UserWarning)#Ignore UserWarningPopup
import pandas as pd
import numpy as np

df = pd.read_csv('Manhattan12.csv') # import csv --> dataframe

print("Initial dataframe shape:", df.shape) # show initial dataframe shape

## Remove excess info at the top, as python recognised that as being the column 
new_headers = df.iloc[3]
for i in range(4):
    df = df.drop(i)
df.columns = new_headers
#------------------------------------#

#Rename incorrect format columns
df = df.rename(columns={'APART\nMENT\nNUMBER':'APARTMENT NUMBER',
                        'SALE\nPRICE':'SALE PRICE'})
#------------------------------------#

#Create list of categorical variables and numerical variables
num_vars = ['BOROUGH','BLOCK','LOT','ZIP CODE', 'RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 
        'TOTAL UNITS', 'LAND SQUARE FEET', 'GROSS SQUARE FEET', 'YEAR BUILT', 'TAX CLASS AT TIME OF SALE',
        'SALE PRICE']

#Remove ',' and '$' from each numerical column if present && convert to numerical value
for cl in num_vars:
    df[cl] = df[cl].str.replace(',', '')
    df[cl] = df[cl].str.replace('$', '')
    df[cl] = df[cl].apply(pd.to_numeric, errors='coerce')
    df[cl] = df[cl].replace(0, np.nan)

#Create numerical variables list
num_vars = list(df.select_dtypes(include=[np.number]).columns)
#Create categorical variables list
categ_vars = list(df.select_dtypes(include=[np.object_]).columns)

#------------------------------------#

#Convert SALE DATE to datetime
df['SALE DATE'] = df['SALE DATE'].apply(pd.to_datetime)
#------------------------------------#


#Remove spaces and replace empty string --> NaN on all categorical variables
for cl in categ_vars:
    df[cl] = df[cl].astype(str)
    df[cl] = df[cl].replace(' ','')
    df[cl] = df[cl].replace('', np.nan)
#------------------------------------#

#Show summary of all missing values as well as summary statistics
missing_values = df.isnull()
summary = missing_values.sum().describe()

print(f"Summary statistics of the missing value:\n{summary}")
#------------------------------------#

#Drop columns 'BOROUGH', 'EASEMENT', 'APARTMENT NUMBER'
df = df.drop(columns=['BOROUGH', 'EASE-MENT', 'APARTMENT NUMBER'])
#Update categorical and numerical variables lists
num_vars.remove('BOROUGH')
categ_vars.remove('EASE-MENT')
categ_vars.remove('APARTMENT NUMBER')


#Drop duplicates if any
df = df.drop_duplicates()

#Drop rows with NaN values
df = df.dropna()


#Identify & remove outliers from numerical variables
for x in num_vars:
    z_scores = np.abs(df[x] - df[x].mean()) / df[x].std()
    df = df[(z_scores < 3)] # z-score of 3 standard

#------------------------------------#

#Log of sale prices
df['LOG OF SALE PRICE'] = np.log(df['SALE PRICE'])

#Normalise data
df_norm = df.select_dtypes(include=[np.number])
df_norm = (df_norm - df_norm.min()) / (df_norm.max() - df_norm.min())
df_norm_desc = df_norm.describe()
#df_norm.to_csv("test_norm.csv", index=False)
#print(df_norm_desc)

#Shape of resulting dataframe
print("Cleaned dataframe shape:", df.shape)


#df.to_csv("test.csv", index=False)
#print(df.head())