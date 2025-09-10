import pandas as  pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('salary_data.csv')

# df.info()

x = df.drop('Salary',axis=1)
y = df['Salary']

encoder = LabelEncoder()

x['Title'] = encoder.fit_transform(x['Title'])

# print(x)

model = LinearRegression()
model.fit(x,y)

output = model.predict(pd.DataFrame([[1,7]],columns=['Title','Experience'])) # 1.Project Manager 2.Software Engineer 0.Data Scientist
print(output)