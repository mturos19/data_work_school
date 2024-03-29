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
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import svm, feature_selection, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor


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
    
df_cln = clean_data(df)


#Normalise data
def normalise_data(cldf):
    cldf_norm = cldf.select_dtypes(include=[np.number])
    cldf_norm = (cldf_norm - cldf_norm.min()) / (cldf_norm.max() - cldf_norm.min())
    print(f"Normalised data summary:\n{cldf_norm.describe()}")
    cldf_norm.to_csv("normalised_data.csv", index=False) # dump normalised data into a .csv
    return cldf_norm

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
viz_pr_ne(df_cln)

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
viz_pr_time(df_cln)

    

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
scat_plot(plot_cols, df_cln)


#Correlation matrix
def corr_plot(plot_cols, df_cln):
    corr_matrix = df_cln[plot_cols].corr()
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=True)
    plt.title('Correlation Matrix')
    plt.savefig("correlation_matrix.jpg")
    plt.show()
corr_plot(plot_cols, df_cln)



# 3. Model building ------------ 

data_norm = normalise_data(df_cln)
df_train, df_test = train_test_split(data_norm, test_size=0.3)
print("Training size: {}, Testing size: {}".format(len(df_train), len(df_test)))
print("Samples: {} Features: {}".format(*df_train.shape))

# Select predictors for model
df_model = data_norm.select_dtypes(include=[np.number]).copy()
feature_cols = df_model.columns.values.tolist()
feature_cols.remove('SALE PRICE')
feature_cols.remove('LOG OF SALE PRICE')
XO = df_model[feature_cols]
YO = df_model['SALE PRICE']
estimator = svm.SVR(kernel="linear")
selector = feature_selection.RFE(estimator, n_features_to_select=5, step=1)
selector = selector.fit(XO, YO)
select_features = np.array(feature_cols)[selector.ranking_ == 1].tolist()
print('Selected features', select_features)

#Linear model
def linearModel(df_model, select_features):
    global X
    X = df_model[select_features]
    global Y
    Y = df_model['SALE PRICE']
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
    global lm
    lm = linear_model.LinearRegression()
    lm.fit(trainX, trainY)
    # Model equations
    print("Y-axis intercept {:6.4f}".format(lm.intercept_))
    print("Weight coefficients:")
    for feat, coef in zip(select_features, lm.coef_):
        print(" {:>20}: {:6.4f}".format(feat, coef))
    # R2 value
    print("R squared for the training data is {:4.3f}".format(lm.score(trainX, trainY)))
    print("Score against test data: {:4.3f}".format(lm.score(testX, testY)))

    #show histogram of residuals
    residuals = Y - lm.predict(X)

    plt.hist(residuals, bins=30)
    plt.title('Histogram of Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.savefig('residuals_histogram.jpeg')
    plt.show()

    #cross validate model
    cv_score = cross_val_score(lm, X, Y, cv=5)
    print("Linear model cross validation score: ", cv_score)

linearModel(df_model, select_features)

#evaluate the mean squared error
def mse(df_model, pred, obs):
    n = df_model.shape[0]
    return sum((df_model[pred]-df_model[obs])**2)/n
df_model['pred'] = lm.predict(X)
print("Mean Squared error: {}".format(mse(df_model,'pred','SALE PRICE')))




# Part 2 - Improved Model
imputer = SimpleImputer(strategy='mean')
imputer.fit(df_model)
df_imputed = pd.DataFrame(imputer.transform(df_model), columns=df_model.columns)


#Decision tree model
def decisiontree(df_model, feature_cols):
    X = df_model[feature_cols]
    Y = df_model['SALE PRICE']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

    tm = DecisionTreeRegressor()
    tm.fit(X_train, Y_train)

    Y_pred = tm.predict(X_test)

    #RMSE
    rmse = mean_squared_error(Y_test, Y_pred, squared=False)
    print("Decision tree root mean squared error:", rmse)

    #Cross validation of decision tree model
    cv_score = cross_val_score(tm, X, Y, cv=5)
    print("Decision tree cross validation score: ", cv_score)


decisiontree(df_model, feature_cols)


def KMeansAlgo(df_imputed):
    
    #Initial histogram
    plt.hist(df_imputed['SALE PRICE'])
    plt.xlabel('SALE PRICE')
    plt.title("Sale Price")
    plt.grid()
    plt.savefig("initial_histogram.jpeg")
    plt.show()

    km = KMeans(n_clusters=5)
    km.fit(df_imputed)
    #J-score
    print('J-score= ', km.inertia_)
    cluster_labels = km.labels_
    md = pd.Series(cluster_labels)
    df_imputed['clust'] = md

    #cluster centers
    centroids = km.cluster_centers_
    print('centroids', centroids)

    #histogram of clusters
    plt.hist(df_imputed['clust'])
    plt.title("Histogram of Clusters")
    plt.xlabel('Cluster')
    plt.ylabel('Frequency')
    plt.grid()
    plt.savefig("cluster_histogram.jpeg")
    plt.show()


    ######## 2D plot of unclustered data
    pca_data = PCA(n_components=2).fit(df_imputed)
    pca_2d = pca_data.transform(df_imputed)
    plt.scatter(pca_2d[:,0], pca_2d[:,1])
    plt.title('Unclustered Data')
    plt.savefig("unclustered_data_plot.jpeg")
    plt.show()

    ######## 2D plot of the clusters
    plt.scatter(pca_2d[:,0], pca_2d[:,1], c=cluster_labels)
    plt.title('Price Clusters')
    plt.savefig("prices_clusters_plot.jpeg")
    plt.show()

    #Regressor on clusters
    clusters = km.fit_predict(X)
    for i in range(km.n_clusters):
        X_cluster = X[clusters == i]
        Y_cluster = Y[clusters == i]

        knn = KNeighborsRegressor(n_neighbors=3) # k=3
        knn.fit(X_cluster, Y_cluster)

        new_test = [0.165, 0.18, 0.108108, 0.1111, 0.23454]
        predicted_price = knn.predict([new_test])

        print(f"Predicted price for cluster {i}: {predicted_price}")



KMeansAlgo(df_imputed)

