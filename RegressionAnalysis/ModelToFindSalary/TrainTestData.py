import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder , OneHotEncoder , MinMaxScaler , StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('SpendingData.csv')
x = data.drop("Spendings",axis=1)
y = data['Spendings']

X_train , X_test , Y_train , Y_test = train_test_split(x,y,test_size=0.2,random_state=1000)

# print("X_train:\n",X_train.head(),"\n")
# print("X_test:\n",X_test.head(),"\n")
# print("Y_train:\n",Y_train.head(),"\n")
# print("Y_Test:\n",Y_Test.head(),"\n")

model = LinearRegression()
model.fit(X_train,Y_train)

train_score = model.score(X_train, Y_train)
test_score = model.score(X_test,Y_test)

print(f"Train Score: {train_score}")
print(f"Test Score: {test_score}")