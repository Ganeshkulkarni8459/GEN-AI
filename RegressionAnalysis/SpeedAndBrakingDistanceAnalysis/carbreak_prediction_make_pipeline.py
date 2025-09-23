import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

df = pd.read_csv('Data1.csv')

corr = df['Speed'].corr(df['BrakingDistance'])

X = df[['Speed']]
Y = df['BrakingDistance']

degree = 2
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

model.fit(X, Y)

output = model.predict(pd.DataFrame([[120]], columns=['Speed']))
print(f"Predicted Braking Distance at 120 Speed: {output[0]}")