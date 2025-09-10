import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv('Data1.csv')

x = df.drop(columns=['BrakingDistance'],axis=1)
y =  df['BrakingDistance']

poly = PolynomialFeatures(degree=5)

x_square = poly.fit_transform(x)

model = LinearRegression()

model.fit(x_square,y)

output =  model.predict(poly.fit_transform([[115]]))

print(output)

#Save Model Into File

with open('car_prediction.bin','wb') as file:
    pickle.dump(model, file)