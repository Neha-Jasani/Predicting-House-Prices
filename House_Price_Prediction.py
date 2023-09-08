#!/usr/bin/env python
# coding: utf-8

# # Problem Statement: House Price Prediction
# * The data contains 1460 training data points and 80 features that might help to predict the selling price of a house.

# In[1]:


# import library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

import warnings
warnings.filterwarnings('ignore')


# # Step 1: Reading and Understanding data

# In[2]:


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


# In[3]:


# Check number of rows and columns
df_train.shape


# In[4]:


df_test.shape


# In[5]:


# Top 5 record
pd.set_option('display.max_columns', 85)
df_train.head()


# In[6]:


df_test.head()


# In[7]:


# Check the summary
df_train.info()


# In[8]:


df_test.info()


# In[9]:


# Statistic of numeric column
df_train.describe()


# In[10]:


df_test.describe()


# # Step 2: Data Cleaning

# ### 1. Checking missing value/treatment of missing value:-

# In[11]:


df_train.isnull().sum()


# In[12]:


# Checking the percentage of null values
print(((df_train.isnull().sum()/len(df_train))*100).sort_values(ascending=False))


# In[13]:


df_train = df_train.drop(['PoolQC','MiscFeature','Alley','Fence'],1)


# ### Observation :-
# * Here missing value in dataframe.
# * Here 'PoolQC','MiscFeature','Alley' and 'Fence' columns more than 50% data are missing so we simply drop that columns.

# In[14]:


df_test.isnull().sum()


# In[15]:


# Checking the percentage of null values
print(((df_test.isnull().sum()/len(df_test))*100).sort_values(ascending=False))


# In[16]:


df_test = df_test.drop(['PoolQC','MiscFeature','Alley','Fence'],1)


# ### Observation :-
# * Here missing value in dataframe.
# * Here 'PoolQC','MiscFeature','Alley' and 'Fence' columns more than 50% data are missing so we simply drop that columns.

# ### 2. Treatment of missing values

# ##### Treatment of missing values for numarical data in train dataset

# In[17]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan , strategy='mean')


# In[18]:


# for numarical data
num_df_train = df_train.loc[:,df_train.dtypes != 'object']


# In[19]:


num_df_train.columns


# In[20]:


num_arr_train = imputer.fit_transform(num_df_train)
df1 = pd.DataFrame(num_arr_train)
df1.head()


# In[21]:


df1.columns = ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
       'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
       'MiscVal', 'MoSold', 'YrSold', 'SalePrice']
df1.head()


# ##### Treatment of missing values for categorical data in train dataset

# In[22]:


cat_imputer = SimpleImputer(missing_values=np.nan , strategy='most_frequent')
cat_df_train = df_train.loc[:,df_train.dtypes == 'object']


# In[23]:


cat_arr_train = cat_imputer.fit_transform(cat_df_train)
df2 = pd.DataFrame(cat_arr_train)


# In[24]:


df2.head()


# In[25]:


cat_df_train.columns


# In[26]:


df2.columns = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']
df2.head()


# In[27]:


df_train = pd.concat([df1,df2],axis=1)


# In[28]:


df_train.head()


# ### Observation :-
# * For numerical data we use 'mean' strategy for missing value treatement.
# * For categorical data we use 'most frequent' strategy for missing value treatmnet.

# ##### Treatment of missing values for numarical data in test dataset

# In[29]:


# for numarical data
num_df_test = df_test.loc[:,df_test.dtypes != 'object']


# In[30]:


num_df_test.columns


# In[31]:


df_train['LotFrontage']


# In[32]:


num_arr_test = imputer.fit_transform(num_df_test)
df3 = pd.DataFrame(num_arr_test)
df3.head()


# In[33]:


df3.columns = ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
       'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
       'MiscVal', 'MoSold', 'YrSold']
df3.head()


# ##### Treatment of missing values for categorical data in test dataset

# In[34]:


cat_df_test = df_test.loc[:,df_test.dtypes == 'object']


# In[35]:


cat_arr_test = cat_imputer.fit_transform(cat_df_test)
df4 = pd.DataFrame(cat_arr_test)


# In[36]:


df4.head()


# In[37]:


cat_df_test.columns


# In[38]:


df4.columns = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']
df4.head()


# In[39]:


df_test = pd.concat([df3,df4],axis=1)


# In[40]:


df_test.head()


# ### Observation :-
# * For numerical data we use 'mean' strategy for missing value treatement.
# * For categorical data we use 'most frequent' strategy for missing value treatmnet.

# ### 2.checking outliers/treatment:-

# In[41]:


## checking outliers using boxplot:-

df_train.plot(kind='box', subplots=True, figsize=(22,20), layout=(10,4))
plt.show()


# ##### Next, we do capping to 99 percentile on numeric column of train dataframe :

# In[42]:


def num(x):
    plt.figure(figsize=(6,6))
    plt.title(x)
    sns.boxplot(df_train[x])
    plt.show()
    return


# In[43]:


for x in num_df_train:
    q3,q1 = np.percentile(df_train[x],[75,25])
    q4= np.percentile(df_train[x],[99])
    df_train.loc[df_train[x] > q4[0], x] = q4[0]
    num(x)


# In[44]:


## checking outliers using boxplot:-

df_test.plot(kind='box', subplots=True, figsize=(22,20), layout=(10,4))
plt.show()


# ##### Next, we do capping to 99 percentile on numeric column of test dataframe :

# In[45]:


def num(x):
    plt.figure(figsize=(6,6))
    plt.title(x)
    sns.boxplot(df_test[x])
    plt.show()
    return


# In[46]:


for x in num_df_test:
    q3,q1 = np.percentile(df_test[x],[75,25])
    q4= np.percentile(df_test[x],[99])
    df_test.loc[df_test[x] > q4[0], x] = q4[0]
    num(x)


# # Step-3. Data analysis 

# In[47]:


num_co = list(df_train.describe().columns)


# In[48]:


for col in num_co:
    plt.figure(figsize=(6,6))
    sns.distplot(df_train[col])
    plt.title(col)
    plt.show()


# # 4.Data Preparation

# ### Encoding

# In[49]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[50]:


cat_col = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']


# In[51]:


for i in cat_col:
    df_train[i]= le.fit_transform(df_train[i])


# In[52]:


df_train.head()


# In[53]:


from sklearn.preprocessing import StandardScaler
scl = StandardScaler()


# ### 2. Build a Simple Linear Regression model to predict the Sale price of the house. Use Area as the independent variable 

# In[54]:


X = df_train['LotArea']
y = df_train['SalePrice']


# In[55]:


# reshape the value of x
x_rshp = X.values.reshape((-1,1))


# In[56]:


scl.fit(x_rshp)


# In[57]:


X = scl.fit_transform(x_rshp)
# y = scl.transform(y)


# In[58]:


X


# In[59]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[60]:


# Train the model 
model.fit(X,y)


# In[61]:


#predict the value of y
y_pred_simple_re = model.predict(X)


# In[62]:


y_pred_simple_re


# In[63]:


from sklearn.metrics import r2_score

r2_score(y,y_pred_simple_re)


# In[64]:


# sns.scatterplot(X,y_rshp)
# plt.plot(X,y_pred_simple_re,'r')


# ## Train_Test_Split

# In[65]:


## CREATE X and y
X = df_train.drop('SalePrice',axis=1)
Y = df_train['SalePrice']


# In[66]:


#### Here we create TRAIN | VALIDATION | TEST  #########
from sklearn.model_selection import train_test_split


# In[67]:


# 70% of data is training data, set aside other 30%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=101)


# ##### Feature selection using RFE :

# In[68]:


from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE


# In[69]:


lr = LinearRegression()
rfe = RFE(estimator = lr,n_features_to_select = 13) # we are finding top 7 features or columns


# In[70]:


rfe = rfe.fit(X,Y)
rfe


# In[71]:


rfe.support_


# In[72]:


rfe.ranking_


# In[73]:


a = zip(tuple(X.columns),tuple(rfe.support_))


# In[74]:


print(tuple(a))


# In[75]:


X_train = X_train.drop(['Id','MSSubClass','LotFrontage','LotArea','OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
                        'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtHalfBath', 'FullBath',
                        'HalfBath', 'BedroomAbvGr','GarageYrBlt', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 
                        'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'MSZoning', 'LotShape', 'LandContour',  'LotConfig',  'Neighborhood', 'Condition1', 
                        'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterCond', 'Foundation',  
                        'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'Electrical',  'Functional', 'FireplaceQu', 'GarageType',
                        'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition'],axis=1)


# In[76]:


X_test = X_test.drop(['Id','MSSubClass','LotFrontage','LotArea','OverallCond','OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
                        'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtHalfBath', 'FullBath',
                        'HalfBath', 'BedroomAbvGr','GarageYrBlt', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 
                        'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'MSZoning', 'LotShape', 'LandContour',  'LotConfig',  'Neighborhood', 'Condition1', 
                        'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterCond', 'Foundation',  
                        'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'Electrical',  'Functional', 'FireplaceQu', 'GarageType',
                        'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition'],axis =1)


# In[77]:


X_train.head()


# In[78]:


X_test.head()


# ### Scale data

# In[79]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[80]:


X_train


# In[81]:


X_test


# ### 3. Build Multiple Linear Regression model to predict Sale price of the house.

# In[82]:


lr_mul_reg = LinearRegression()


# In[83]:


lr_mul_reg.fit(X_train,y_train)


# In[84]:


y_pred_mul_reg = lr_mul_reg.predict(X_test)


# In[85]:


r2_score(y_test,y_pred_mul_reg)


# ### 4. Use dimensionality reduction technique PCA/LDA and build Multiple Linear Regression model to predict Sale price of the house.

# ###### Use dimensionality reduction technique LDA

# In[86]:


## CREATE X and y
X = df_train.iloc[:,:-1].values
Y = df_train.iloc[:, 1].values


# In[87]:


scl_lda = StandardScaler()


# In[88]:


X = scl.fit_transform(X)


# In[89]:


# 70% of data is training data, set aside other 30%
X_train_lda, X_test_lda, y_train_lda, y_test_lda = train_test_split(X, Y, test_size=0.2, random_state=101)


# In[90]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# In[91]:


lda = LDA(n_components=2)


# In[92]:


X_train_lda = lda.fit_transform(X_train_lda, y_train_lda)
X_test_lda = lda.transform(X_test_lda)


# In[93]:


lda.intercept_


# In[94]:


lr_lda = LinearRegression()


# In[95]:


lr_lda.fit(X_train_lda,y_train_lda)


# In[96]:


y_pred_lda = lr_lda.predict(X_test_lda)


# In[97]:


r2_score(y_test_lda,y_pred_lda)


# ### 5. Build a model using Lasso and Ridge regression to reduce model complexity.

# ##### Lasso Regression

# In[98]:


from sklearn.linear_model import Lasso


# In[99]:


lasso = Lasso(alpha=0.0001)


# In[100]:


lasso.fit(X_train,y_train)


# In[101]:


y_pred_lasso = lasso.predict(X_test)


# In[102]:


mse = mean_squared_error(y_test,y_pred_lasso)
mse


# In[103]:


lasso.coef_


# In[104]:


r2_score(y_test,y_pred_lasso)


# ##### Ridge Regression

# In[105]:


from sklearn.linear_model import Ridge


# In[106]:


ridge = Ridge(alpha=0.0001)


# In[107]:


ridge.fit(X_train,y_train)


# In[108]:


y_pred_ridge = ridge.predict(X_test)


# In[109]:


mse = mean_squared_error(y_test,y_pred_ridge)
mse


# In[110]:


r2_score(y_test,y_pred_ridge)


# ### 6. Build SVR model to predict Sale price of the house.

# In[111]:


from sklearn.svm import SVR


# In[112]:


svr = SVR(C = 10000 , gamma = 0.2)


# In[113]:


svr.fit(X_train,y_train)


# In[114]:


svr_pred = svr.predict(X_test)


# In[115]:


r2_score(y_test,svr_pred)


# ### 7. Build Decision Tree Regressor to predict Sale price of the house.

# In[116]:


from sklearn.tree import DecisionTreeRegressor


# In[117]:


#Create Decision Tree Classifer for criterion="gini"
dt = DecisionTreeRegressor()


# In[118]:


# fit the model
dt.fit(X_train,y_train)


# In[119]:


y_pred_tree = dt.predict(X_test)


# In[120]:


r2_score(y_test,y_pred_tree)


# ### 8. Build Random Forest Regression model to predict Sale price of the house.

# In[121]:


from sklearn.ensemble import RandomForestRegressor


# In[122]:


rfr = RandomForestRegressor()


# In[123]:


rf_regressor = rfr.fit(X_train,y_train)


# In[124]:


y_pred_rfr = rf_regressor.predict(X_test)


# In[125]:


r2_score(y_pred_rfr,y_test)


# ### 9. Use GridsearchCV and RandomizedsearchCV for tuning hyperparameters and fit your model on the optimal              parameters. 

# ##### GridsearchCV

# In[126]:


from sklearn.model_selection import GridSearchCV


# In[127]:


rfr = RandomForestRegressor()


# In[128]:


param_grid = { 
            "n_estimators"      : [10,20,30],
            "max_features"      : ["auto", "sqrt", "log2"],
            "min_samples_split" : [2,4,8],
            "bootstrap": [True, False],
            }


# In[129]:


grid = GridSearchCV(rfr, param_grid, n_jobs=-1, cv=5)


# In[130]:


grid.fit(X_train, y_train)


# In[131]:


print('Best hyper parameter :' , grid.best_params_)


# In[132]:


rfr = RandomForestRegressor(bootstrap=True , max_features= 'sqrt' , n_estimators= 30 , min_impurity_split=8)


# In[133]:


rf_grid = rfr.fit(X_train,y_train)


# In[134]:


y_pred_rfr_grid = rf_grid.predict(X_test)


# In[135]:


r2_score(y_pred_rfr_grid,y_test)


# ##### RandomizedsearchCV

# In[136]:


from sklearn.model_selection import RandomizedSearchCV


# In[137]:


random_grid = {'bootstrap': [True, False],
               'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
               'max_features': ['auto', 'sqrt'],
               'min_samples_leaf': [1, 2, 4],
               'min_samples_split': [2, 5, 10],
               'n_estimators': [130, 180, 230]}


# In[138]:


rf = RandomForestRegressor()


# In[139]:


rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100,
                               cv = 3, verbose=2, random_state=42, n_jobs = -1)


# In[140]:


rf_random.fit(X_train,y_train)


# In[141]:


print('Best hyper parameter :' , grid.best_params_)


# In[142]:


rf_random.best_score_


# In[143]:


rf = RandomForestRegressor(bootstrap=True , max_features= 'sqrt' , n_estimators= 20 , min_impurity_split=8)


# In[144]:


rf_randomized = rf.fit(X_train,y_train)


# In[145]:


y_pred_randomized = rf_randomized.predict(X_test)


# In[146]:


r2_score(y_test,y_pred_randomized)


# ### 10. Model Selection: Evaluate and compare performance of all the models to find the best model. 

# In[147]:


compare = pd.DataFrame({'Model':['Simple Linear Regression' ,' Multiple Linear Regression' , 'Lasso' , 'Ridge' ,'SVR',
                                'Decision Tree Regressor' , 'Random Forest Regression'],
                        'r2_score':[r2_score(y,y_pred_simple_re)*100,r2_score(y_pred_mul_reg,y_test)*100,
                                    r2_score(y_test,y_pred_lasso)*100,r2_score(y_test,y_pred_ridge)*100,
                                    r2_score(y_test,svr_pred)*100,r2_score(y_test,y_pred_tree)*100,
                                    r2_score(y_test,y_pred_rfr)*100]})


# In[148]:


compare.sort_values(by='r2_score', ascending=False)

