# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:23:21 2020
#author: Nikita Porwal
#WhatsApps - +91-9424847351
#Email: nikita.porwal05@gmail.com
#Linkedin: https://linkedin.com/in/nikita-porwal
"""

#####LINEAR REGRESSION####
'''
Regression has the following assumptions:

1. Linear relationship
2. Multivariate normality
3. No or little multicollinearity
4. No auto-correlation
5. Homoscedasticity (all variables have same variance)
'''

#Import Required Packages
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns


#Import The Data
train = pd.read_csv(r"C:\Users\anish sharma\Documents\Tutions\Imarticus\Projects\Data\Property_Price_Train.csv")
test = pd.read_csv(r"C:\Users\anish sharma\Documents\Tutions\Imarticus\Projects\Data\Property_Price_Test.csv")


#Basic Data Sanity Check
train.shape
test.shape

train.columns
test.columns


#Making Columns Similar to clean both Test and Train data in 1 go
print(train.columns.difference(test.columns))
#We can see that 'Sale_Price' is the column which is extra
    #We will remove Sales_Price Column

train.drop(['Sale_Price'],axis = 1,inplace = True)


#Creating a column to differenciate between Train and Test
train['Sample'] = 'train'
test['Sample'] = 'test'

data = pd.concat([train,test],axis=0,sort=False) #Combine the data

#Check the Data
data.head()
data.info()

#We would need certain functions again and again to audit our data, lets
#create a package called Data_Auditor, which will tell us about the NA in
#Data , NA in column's and also help us perform some basic data manipulation

class Data_Auditor:
    def NA_in_Data(data_frame):
        result = (data_frame.isnull().sum().sum()) / (data_frame.shape[0] * data_frame.shape[1]) * 100
        return(print("Data has",round(result,2),"% NA's"))
    
    def Remove_Columns(data_frame,*args):
        list_of_cols = list(args)
        data_frame.drop(list_of_cols,axis = 1,inplace = True)
        
    def NA_in_Columns(data_frame):
        total_missing = data_frame.isnull().sum().sort_values(ascending=False)
        percent_missing = round(((data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)*100),1)
        missing_data = pd.concat([total_missing, percent_missing], axis=1, keys=['Missing_Obs', 'Percent_of_NA'])
        return(missing_data.head(10))
        
    def Most_Frequent_Data(Data_frame,Column):
        Count = Data_frame[Column].value_counts()
        Percentage = round(((Data_frame[Column].value_counts()/Data_frame.shape[0])*100),2)
        Summary_data = pd.concat([Count, Percentage], axis=1, keys=['Count','Percentage'])
        return(Summary_data)



Data_Auditor.NA_in_Data(data) #There are a lot of NA's we need to investigate further
Data_Auditor.NA_in_Columns(data) #Lets look at it columns wise


##IMPUTING OUT THE MISSING DATA##
#As per Data Description, NA in below mentioned columns
#means these features are not there in the house so lets
#impute it like wise
data["Pool_Quality"] = data["Pool_Quality"].fillna("No pool")
data["Miscellaneous_Feature"] = data["Miscellaneous_Feature"].fillna("No misc feature")
data["Lane_Type"] = data["Lane_Type"].fillna("No alley access")
data["Fence_Quality"] = data["Fence_Quality"].fillna("No fence")
data["Fireplace_Quality"] = data["Fireplace_Quality"].fillna("No fireplace")


#Lot_Extent description says  its Linear feet of
#street connected to property most likely have a
#similar area to other houses in its neighborhood,
#we can fill in missing values by the median
#LotFrontage of the neighborhood.

data.groupby('Neighborhood')['Lot_Extent'].median()
data['Lot_Extent'] = data['Lot_Extent'].fillna(data.groupby('Neighborhood')['Lot_Extent'].transform('median'))

#Garage:NA means the property has no garage hence, we
#can impute other garage related columns with no garage 
Garage_cols = [columns for columns in data if columns.startswith('Garage')]

for columns in Garage_cols:
    if data[columns].dtypes == 'float64':
        data[columns] = 0
    else:
        data[columns] = 'No_Garage'

del Garage_cols 
    
#Basement:NA means the property has no basement hence, we
#can impute other garage related columns with no garage
#additionally Exposure_Level Refers to walkout or garden level walls
#and NA in this situation means no basement

Basement_cols_data = data.loc[:,data.columns.str.contains("Basement")]
Basement_cols_data_2 = data.loc[:,data.columns.str.contains("Bsmt")]
Basement_cols_data_3 = data['Exposure_Level']
Basement = pd.concat([Basement_cols_data, Basement_cols_data_2, Basement_cols_data_3], axis=1)

del Basement_cols_data
del Basement_cols_data_2
del Basement_cols_data_3

for columns in Basement.columns:
    if data[columns].dtypes == 'float64':
        data[columns] = data[columns].fillna(0)
    else:
        data[columns] = data[columns].fillna('No_Basement')

del Basement      
del columns
        
# MasVnrArea and MasVnrType : NA most likely means no Brick
# veneer for these houses. We can fill 0 for the area and None
# for the type.
data["Brick_Veneer_Type"] = data["Brick_Veneer_Type"].fillna("None")
data["Brick_Veneer_Area"] = data["Brick_Veneer_Area"].fillna(0)

#Zoning Class
Data_Auditor.Most_Frequent_Data(data,'Zoning_Class')
#We can observe that RLD has almost 80% data so we can impute it
data['Zoning_Class'] = data['Zoning_Class'].fillna(data['Zoning_Class'].mode()[0])


#Utility Type
Data_Auditor.Most_Frequent_Data(data,'Utility_Type')
#When we read the data type we realise that almost all
#houses have all public utilities there is just one
#house that has no Water (Septic Tank) more or less
#this is constant and hence we can delete this
#feature
Data_Auditor.Remove_Columns(data,'Utility_Type')


#Underground_Full_Bathroom & Underground_Half_Bathroom
Data_Auditor.Most_Frequent_Data(data,'Underground_Full_Bathroom')
data['Underground_Full_Bathroom'] = data['Underground_Full_Bathroom'].fillna(data['Underground_Full_Bathroom'].mode()[0])

Data_Auditor.Most_Frequent_Data(data,'Underground_Half_Bathroom')
data['Underground_Half_Bathroom'] = data['Underground_Half_Bathroom'].fillna(data['Underground_Half_Bathroom'].mode()[0])

#Since both of them have only 2 NA's each we can safely
#impute it with the mode

#Functional_Rate
Data_Auditor.Most_Frequent_Data(data,'Functional_Rate')
data['Functional_Rate'] = data['Functional_Rate'].fillna(data['Functional_Rate'].mode()[0])
#We have used mode imputation again

#Sale_Type
Data_Auditor.Most_Frequent_Data(data,'Sale_Type')
data['Sale_Type'] = data['Sale_Type'].fillna(data['Sale_Type'].mode()[0])

#Electrical_System
Data_Auditor.Most_Frequent_Data(data,'Electrical_System')
data['Electrical_System'] = data['Electrical_System'].fillna(data['Electrical_System'].mode()[0])

#Exterior1st & Exterior2nd
Data_Auditor.Most_Frequent_Data(data,'Exterior1st')
data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])

Data_Auditor.Most_Frequent_Data(data,'Exterior2nd')
data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])

#Kitchen_Quality
Data_Auditor.Most_Frequent_Data(data,'Kitchen_Quality')
data['Kitchen_Quality'] = data['Kitchen_Quality'].fillna(data['Kitchen_Quality'].mode()[0])

#Checking the Data Quality Now
Data_Auditor.NA_in_Data(data)
#We can proceed with next activity now 

#ID is a useless column anyways
Data_Auditor.Remove_Columns(data,'Id')


#FEATURE ENGINEERING
#Since total SqFt is an important feature  in selecting house we add them up
data['Total_HouseArea_SqFt'] = data['Total_Basement_Area'] + data['First_Floor_Area'] + data['Second_Floor_Area']

#DATA CONVERSION
#Encoding some categorical variables into Ordered Numeric Variable as they are ranked
#variables

from sklearn.preprocessing import LabelEncoder

columns = ('Fireplace_Quality','Exterior_Material','BsmtFinType2',
        'Property_Shape','Year_Sold','Basement_Height','Basement_Condition',
        'Garage_Quality','Garage_Condition','Exterior_Condition','Heating_Quality',
        'Pool_Quality','Kitchen_Quality','BsmtFinType1','Functional_Rate','Fence_Quality',
        'Exposure_Level','Garage_Finish_Year','Property_Slope','Pavedd_Drive','Road_Type',
        'Lane_Type','Air_Conditioning','Building_Class','House_Condition',
        'Month_Sold')

# process columns, apply LabelEncoder to categorical features

for cols in columns:
    lbl = LabelEncoder() 
    lbl.fit(list(data[cols].values)) 
    data[cols] = lbl.transform(list(data[cols].values))


#Dummy Value Encoding
data = pd.get_dummies(data)
print(data.shape)

#Split Test and Train Data
#Train
Train_data = data[data['Sample_train'] == 1]
del Train_data['Sample_train']
del Train_data['Sample_test']
df = pd.read_csv(r"C:\Users\anish sharma\Documents\Tutions\Imarticus\Projects\Data\Property_Price_Train.csv")
Train_data = pd.concat([Train_data, df['Sale_Price']], axis=1).reindex(Train_data.index)

#Lets check for outliers
fig, ax = plt.subplots()
ax.scatter(x = Train_data['Total_HouseArea_SqFt'], y = Train_data['Sale_Price'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('SqFt', fontsize=13)
plt.show()

#Removing Outliers
Train_data = Train_data.drop(Train_data[(Train_data['Sale_Price']<200000) &
                                        (Train_data['Total_HouseArea_SqFt']>7500)].index)


#Test
Test_data = data[data['Sample_test'] == 1]
del Test_data['Sample_test']
del Test_data['Sample_train']


#Creating Train Test Data Split for Model 
Input_Train = Train_data[Train_data.columns[Train_data.columns != 'Sale_Price']]
Output_Train = Train_data[Train_data.columns[Train_data.columns == 'Sale_Price']]


#First Model ----> Simple Linear Regression

from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(Input_Train,Output_Train)
r_sq = model.score(Input_Train,Output_Train) #R-Square
B0 = model.intercept_
B1_to_Bk = model.coef_

#Prediction
Output_Train = model.predict(Input_Train)
Output_Test = model.predict(Test_data)

#Finding MAE
from sklearn.metrics import mean_absolute_error
mean_absolute_error(Train_data['Sale_Price'],Output_Train)



#Pro Model -----> Ridge Model
from sklearn.linear_model import Ridge,Lasso
from sklearn.model_selection import GridSearchCV

lambdas=np.linspace(1,100,100)
params={'alpha':lambdas}
ridge_model = Ridge(fit_intercept=True)
grid_search = GridSearchCV(ridge_model,param_grid=params,cv=10,
                           scoring='neg_mean_absolute_error')
grid_search.fit(Input_Train,Output_Train)
grid_search.best_estimator_
grid_search.cv_results_
grid_search.best_estimator_.score(Input_Train,Output_Train) #R-Square

#Prediction
ridge_model_output_train = grid_search.best_estimator_.predict(Input_Train)
ridge_model_output = grid_search.best_estimator_.predict(Test_data)

#Finding MAE
mean_absolute_error(Train_data['Sale_Price'],ridge_model_output_train)



##Pro Model -----> Lasso Model
lambdas=np.linspace(0.001,0.1,100)
lasso_model = Lasso(fit_intercept=True)
params = {'alpha':lambdas}
grid_search = GridSearchCV(lasso_model,param_grid=params,cv=10,
                           scoring='neg_mean_absolute_error')
grid_search.fit(Input_Train,Output_Train)
grid_search.best_estimator_
grid_search.best_estimator_.score(Input_Train,Output_Train) #R-Square

#Prediction
lasso_model_output_train = grid_search.best_estimator_.predict(Input_Train)
lasso_model_output = grid_search.best_estimator_.predict(Test_data)

#Finding MAE
mean_absolute_error(Train_data['Sale_Price'],lasso_model_output_train)

