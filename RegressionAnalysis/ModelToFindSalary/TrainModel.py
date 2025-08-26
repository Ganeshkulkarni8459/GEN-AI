import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('salary_data.csv')

x = df.drop('Salary', axis=1)
y = df['Salary']

model = LinearRegression()

model.fit(x, y)

salaries = model.predict(pd.DataFrame([[15],[30]], columns=['Experience']))

print("Predicted salary for 15 years of experience:", salaries[0])
print("Predicted salary for 30 years of experience:", salaries[1])

print("Model Coefficient (m):", model.coef_[0])
print("Model Intercept (c):", model.intercept_)

m = model.coef_[0]
c = model.intercept_
salary = m * 30 + c
print("Calculated salary for 15 years of experience using y=mx+c:", salary)