import pandas as pd 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split

df = pd.read_csv("C:\\Users\\lenovo\\Downloads\\ML_GIT_Workspace\\ML-Learning\\Datasets\\rawData\\Order_delivery.csv")
print(df.head())
print(df.info())
print(df.columns)
df2 = df[["Order_ID","Delivery_Duration_Minutes","Delivery_Distance_km","Traffic_Level"]]
print(df2.head())
print(df2.info())
df4 = df2[["Traffic_Level"]]
# apply one hot encoding as traffic level is in the str
encoder = OneHotEncoder(drop="first",handle_unknown="ignore")
df3 = encoder.fit_transform(df2[["Traffic_Level"]]).toarray()
print(df3)
encoded_df=pd.DataFrame(df3,columns=encoder.get_feature_names_out())
#df5=pd.concat([df2.drop(["Traffic_Level"]),encoded_df],axis=1)
df5=pd.concat([df2.drop(["Traffic_Level"],axis=1),encoded_df],axis=1)
print(df5.head())
print(df5.info())
# feature scaling for the data 
sclr = StandardScaler()
dfsclr = sclr.fit_transform(df5)
print(dfsclr)
# dfsclar returns a numpy array

print("demo",dfsclr.shape)
# to check the multi-colinearity 
vif = pd.DataFrame();
vif ["Features"]=df5.columns
vif["VIF"] = [variance_inflation_factor(dfsclr,i) for i in range(dfsclr.shape[1])]

print(vif)
# all looks good

#X_train,X_test,y_train,y_test=train_test_split(dfsclr,y,test_size=0.2,random_state=1) \








