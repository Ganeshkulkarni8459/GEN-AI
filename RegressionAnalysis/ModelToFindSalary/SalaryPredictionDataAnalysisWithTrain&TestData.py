import pandas as pd
import  numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

df = pd.read_csv('salary_data.csv')

plt.scatter(df['Experience'],df['Salary'])
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.title('Experience VS Salary')
# plt.show()

x = df.drop('Salary',axis=1)
y = df['Salary']

model = LinearRegression();
model.fit(x,y)

re_predict_all_x = model.predict(x)

output = model.predict([[11]])
print(output)

# Since we have model which is trained , we can plot best fit regrssion line
plt.scatter(df['Experience'],df['Salary'],label = 'Observed Data')
plt.scatter(df['Experience'], re_predict_all_x,color='blue',label='Predicted Data')
plt.plot(df['Experience'],re_predict_all_x,color='red',label='Best Fit Line')

score = model.score(x,y)
print(f"Score: {score}")

plt.show()
# Testing Data
# 2,43525
# 3.2,54445
# 4.9,67938
# 8.7,109431
# 12,155000
