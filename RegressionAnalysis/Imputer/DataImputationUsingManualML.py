import pandas as pd
from sklearn.linear_model import LinearRegression
df = pd.read_csv('salary_data_manual_ml.csv')

# df.info()

x = df.drop('Salary',axis=1)
y = df['Salary']

model = LinearRegression()
model.fit(x,y)

salaries = model.predict(pd.DataFrame([[2],[6]],columns=['Experience']))

# Removing Nan Values
# print("Predicted salary for 2 years of experience:", salaries[0])
# print("Predicted salary for 6 years of experience:", salaries[1])

# Predicted salary for 2 years of experience: 43166.08934786429
# Predicted salary for 6 years of experience: 83213.1780309969

# Again Adding Nan Values after manual prediction

print("Predicted salary for 2 years of experience:", salaries[0])
print("Predicted salary for 6 years of experience:", salaries[1])

# Output
# Predicted salary for 2 years of experience: 43166.077556466116
# Predicted salary for 6 years of experience: 83213.17000272943


