import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('hearing_test.csv')

# print(df.corr())


hearing_problem = df[df['test_result']==1]
no_hearing_problem = df[df['test_result']==0]

plt.scatter(hearing_problem['age'],hearing_problem['physical_score'],color="red",marker=".")
plt.scatter(no_hearing_problem['age'],no_hearing_problem['physical_score'],color="green",marker=".")
# plt.show()

X = df.drop('test_result',axis=1)
Y = df['test_result']

model = LogisticRegression()

model.fit(X,Y)

output = model.predict(pd.DataFrame([[30,28]],columns=['age','physical_score']))
print(output)