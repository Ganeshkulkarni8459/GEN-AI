import pandas as pd
import matplotlib.pyplot as py
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data_frame = pd.read_csv('ModifiedDataSet.csv')

x = data_frame.drop('BrakingDistance',axis=1)
y = data_frame['BrakingDistance']

poly = PolynomialFeatures(degree=2)  #modified dataset -- degree 5
x_square = poly.fit_transform(x)

model = LinearRegression()
model.fit(x_square,y)

output = model.predict(poly.fit_transform([[115]]))
output1 = model.predict(poly.fit_transform([[120]]))
output2 = model.predict(poly.fit_transform([[85]]))
# output3 = model.predict(poly.fit_transform([[55]]))
output4 = model.predict(poly.fit_transform([[65]]))

output4 = model.predict(poly.fit_transform([[65]]))

print(output)
print(output1)
print(output2)
print(output4)

# print(data_frame)

# print(data_frame.info())

# print(data_frame.describe())

# correlation = data_frame['Speed'].corr(data_frame['BrakingDistance'])

# print(correlation)

# py.plot(data_frame['Speed'],data_frame['BrakingDistance'])
# py.xlabel('Speed')
# py.ylabel('BrakingDistance')
# py.title('Speed Vs BrakingDistance')
# py.show()