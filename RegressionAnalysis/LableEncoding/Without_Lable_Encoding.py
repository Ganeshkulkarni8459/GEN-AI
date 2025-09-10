import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Without_lable_encoding.csv')

x = df.drop('Salary',axis=1)
y = df['Salary']

model = LinearRegression()

model.fit(x,y)

output = model.predict(pd.DataFrame([[1,7]],columns=['Title','Experience'])) # 1.Project Manager 2.Software Engineer 0.Data Scientist

print(output)