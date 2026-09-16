import pandas as pd 
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import numpy as np

df = pd.read_excel("C:\\Users\\lenovo\\Downloads\\ML_GIT_Workspace\\ML-Learning\\Datasets\\rawData\\fetch_california_housing.xlsx")

X = df.drop(columns=["price"])
y = df["price"]
#Checking multi colinerity

vif = pd.DataFrame()
vif["Features"]=X.columns 
vif["VIF"]= [variance_inflation_factor(X,i) for i in range(X.shape[1])]

print(vif)
#Deleting columns having  multi colinerarity
X=X[['MedInc', 'HouseAge', 'AveRooms', 'Population', 'AveOccup','Longitude']]

sclr = StandardScaler()

X_sclr = sclr.fit_transform(X)
print(X_sclr)

print(X_sclr.shape)

X_train,X_test,y_train,y_test=train_test_split(X_sclr,y,test_size=0.2,random_state=1)
#Random state -1 is the pointer, to split the data and maintain the same pointer 

print(X_train.shape,X_test.shape)

modal = Ridge()
modal.fit(X_train,y_train)

print("Modal_coef",modal.coef_)
print("modal intercept",modal.intercept_)

y_pred = modal.predict(X_test)
print(y_pred)
print(y_test)

#evalution matrixs
#mean squred error------->
#mean absolute error
#Root  mean squered error
#R squre

print("mse->",mean_absolute_error(y_test,y_pred))
print("mse->",mean_squared_error(y_test,y_pred))
print("rsme->",np.sqrt(mean_absolute_error(y_test,y_pred)))
print("R squre->",r2_score(y_test,y_pred))
