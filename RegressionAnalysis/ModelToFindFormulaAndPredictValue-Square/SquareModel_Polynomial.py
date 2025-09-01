import pandas as pd
import matplotlib.pyplot as ply
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('data.csv')

x = df.drop(columns=['b'],axis=1)
y = df['b']

poly = PolynomialFeatures(degree=2)
x_square = poly.fit_transform(x)

model = LinearRegression()
model.fit(x_square,y)

output = model.predict(poly.fit_transform([[617]]))
print(output)