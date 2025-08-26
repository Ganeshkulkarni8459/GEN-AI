import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

data_frame = pd.read_csv('salary_data.csv')

print(data_frame.columns)

data_frame.info()

print(data_frame.describe())

# plt.scatter(data_frame['Experience'], data_frame['Salary'])
# plt.xlabel('Experience')
# plt.ylabel('Salary')
# plt.title('Experience vs Salary')
# plt.show()

correlation = data_frame['Experience'].corr(data_frame['Salary'])
print(f'Correlation between Experience and Salary: {correlation}')
covariance = data_frame['Experience'].cov(data_frame['Salary'])
print(f'Covariance between Experience and Salary: {covariance}')

print("Mean Salary:", data_frame['Salary'].mean())
print("Median Salary:", data_frame['Salary'].median())
print("Mode Salary:", data_frame['Salary'].mode()[0])
print("Standard Deviation of Salary:", data_frame['Salary'].std())
print("Variance of Salary:", data_frame['Salary'].var())