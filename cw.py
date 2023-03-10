# Mark Turos
# 199015662
# CO3095



## Part 1 - Building a basic predictive model

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)#Ignore FutureWarningPopup
warnings.simplefilter(action='ignore', category=UserWarning)#Ignore UserWarningPopup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns

df = pd.read_csv('Manhattan12.csv') # import csv --> dataframe
print("Initial dataframe shape:", df.shape) # show initial dataframe shape


#Clean the data
def clean_data(df):
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

    #Remove spaces and replace empty string --> NaN on all categorical variables
    for cl in categ_vars:
        df[cl] = df[cl].astype(str)
        df[cl] = df[cl].str.replace(' ','')
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


    #Convert SALE DATE to datetime
    df['SALE DATE'] = pd.to_datetime(df['SALE DATE'])
    
    #------------------------------------#


    #Log of sale prices
    df['LOG OF SALE PRICE'] = np.log(df['SALE PRICE'])

    #Shape of resulting dataframe
    print("Cleaned dataframe shape:", df.shape)
    df.to_csv("clean_data.csv", index=False) # dump data to csv
    return df
    

clean_data(df)
df_cln = clean_data(df)


#Normalise data
def normalise_data(cldf):
    cldf_norm = cldf.select_dtypes(include=[np.number])
    cldf_norm = (cldf_norm - cldf_norm.min()) / (cldf_norm.max() - cldf_norm.min())
    cldf_norm_desc = cldf_norm.describe()
    print(f"Normalised data summary:\n{cldf_norm_desc}")
    cldf_norm.to_csv("normalised_data.csv", index=False) # dump normalised data into a .csv
    return cldf_norm
#normalise_data(df_cln)


# 2. Data exploration ------------ 


#Visualize the prices across neighborhood
#Using heatmap due to sheer amount of volume, making other forms unintelligible
def viz_pr_ne(df):
    neigh_prices = df.groupby('NEIGHBORHOOD')['SALE PRICE'].mean().round()
    neigh_df = pd.DataFrame({'NEIGHBORHOOD':neigh_prices.index, 'SALE PRICE':neigh_prices.values})
    price_neigh_matrix = neigh_df.pivot(index='NEIGHBORHOOD', columns='SALE PRICE', values='SALE PRICE')
    sns.heatmap(price_neigh_matrix, cmap='coolwarm')
    plt.tight_layout()
    plt.savefig("heatmap_price_neighborhood.jpg")
    plt.show()
#viz_pr_ne(df_cln)

#Visualize prices over time, line chart
def viz_pr_time(df):
    df['year'] = df['SALE DATE'].dt.year
    yearly_prices = df.groupby('year')['SALE PRICE'].mean()
    plt.plot(yearly_prices.index, yearly_prices.values)
    plt.xlabel('Year')
    plt.ylabel('Prices')
    plt.title('Prices over time')
    plt.savefig('prices_time.jpg')
    plt.show()
#viz_pr_time(df_cln)

    

#Scatter matrix
plot_cols = list(df_cln.select_dtypes(include=[np.number]))
def scat_plot(plot_cols, df):
    fig = plt.figure(1, figsize=(24,28))
    fig.clf()
    ax = fig.gca()
    scatter_matrix(df[plot_cols], diagonal='hist', ax=ax)
    plt.title("Scatter Matrix")
    plt.savefig("scatter_matrix.jpg", dpi=300)
    plt.show()
#scat_plot(plot_cols, df_cln)


#Correlation matrix
def corr_plot(plot_cols, df_cln):
    corr_matrix = df_cln[plot_cols].corr()
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=True)
    plt.title('Correlation Matrix')
    plt.savefig("correlation_matrix.jpg")
    plt.show()
#corr_plot(plot_cols, df_cln)



# 3. Model building ------------ 

#Predictors: Gross Square Feet, Land Square Feet, Sale Date, 