import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

df = pd.read_csv('salary_data.csv')

# Simple Imputer
# Mean , Median ,  Constant, most_frequent
# imputer = SimpleImputer(strategy='most_frequent')
# imputer = SimpleImputer(strategy='constant',fill_value=21222)
imputer = SimpleImputer(strategy='mean')


df['Salary']  = imputer.fit_transform(df[['Salary']])

print(df)

# x = df.drop('Salary',axis=1)
# y = df['Salary']

# poly = PolynomialFeatures(degree=2)
# x_square = poly.fit_transform(x)

# model = LinearRegression()
# model.fit(x_square,y)

# predictionFor2YearExp = model.predict(poly.fit_transform([[2],[15]]))

# print(predictionFor2YearExp)