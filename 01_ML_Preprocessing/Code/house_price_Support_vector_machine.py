import pandas as pd 
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,root_mean_squared_error
df = pd.read_excel("C:\\Users\\lenovo\\Downloads\\ML_GIT_Workspace\\ML-Learning\\Datasets\\rawData\\fetch_california_housing.xlsx")
print(df.corr)

X=df.drop(columns=["price"])
y= df["price"]

vif = pd.DataFrame()
vif["Features"]= X.columns
vif["vif"] = [variance_inflation_factor(X,i) for i in range(X.shape[1])]

X=X[['MedInc', 'HouseAge', 'AveRooms', 'Population', 'AveOccup','Longitude']]
print(vif)

sclr= StandardScaler()
X_Sclr= sclr.fit_transform(X)

print(X_Sclr)

X_train,X_test,y_train,y_test= train_test_split(X_Sclr,y,test_size=0.2,random_state=1)

model= SVR()
model.fit(X_train,y_train)


ls_y_pred = model.predict(X_test)


#evalution matrix
print("mse->",mean_absolute_error(y_test,ls_y_pred))
print("mse->",mean_squared_error(y_test,ls_y_pred))
print("rsme->",np.sqrt(mean_absolute_error(y_test,ls_y_pred)))
print("R squre->",r2_score(y_test,ls_y_pred))



