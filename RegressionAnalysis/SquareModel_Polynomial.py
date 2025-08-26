import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

try:
    df = pd.read_csv('data.csv')
except FileNotFoundError:
    print("Error: data.csv not found!")
    exit()

if 'a' not in df.columns or 'b' not in df.columns:
    print("Error: Columns 'a' or 'b' not found in dataset")
    exit()
print("############# Plot Graph : ")
plt.scatter(df['a'], df['b'])
plt.xlabel('a - Number')
plt.ylabel('b - Square of a')
plt.title('Number vs Square of Number')
# plt.show() 

model = LinearRegression()

X = df[['a']]  
y = df['b']

poly = PolynomialFeatures(degree=2,include_bias=False)  # Since b = a², we use degree=2
a_poly = poly.fit_transform(df[['a']])

model.fit(a_poly, df['b'])

a_test = poly.transform([[12]])
square_predict = model.predict(a_test)
print("Square of 12 is:", square_predict[0])