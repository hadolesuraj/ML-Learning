import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,root_mean_squared_error
df = pd.read_excel("C:\\Users\\lenovo\\Downloads\\ML_GIT_Workspace\\ML-Learning\\Datasets\\rawData\\fetch_california_housing.xlsx")

print(df.corr)
X = df.drop(columns= ["price"])
y = df["price"]

vif = pd.DataFrame()
vif["Features"]= X.columns
vif["vif"]= [variance_inflation_factor(X,i) for i in range(X.shape[1])]

print(vif)

X=X[['MedInc', 'HouseAge', 'AveRooms', 'Population', 'AveOccup','Longitude']]

sclr= StandardScaler()

X_sclr = sclr.fit_transform(X)

print(X_sclr)

X_train,X_test,y_train,y_test = train_test_split(X_sclr,y,test_size=0.2,random_state=1)

model = DecisionTreeRegressor()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("mse->",mean_absolute_error(y_test,y_pred))
print("mse->",mean_squared_error(y_test,y_pred))
print("rsme->",np.sqrt(mean_absolute_error(y_test,y_pred)))
print("R squre->",r2_score(y_test,y_pred))






