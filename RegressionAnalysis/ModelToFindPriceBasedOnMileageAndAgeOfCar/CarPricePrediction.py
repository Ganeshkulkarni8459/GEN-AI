import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('car_price_data.csv')

x = df.drop('Price',axis=1)

y = df['Price']

model = LinearRegression()

model.fit(x,y)

price_of_car = model.predict(pd.DataFrame([[7,16394],[4,74092]],columns=['Age','Mileage']))

print("Predicted Price for Bike Age=7, Mileage=16394:",price_of_car[0])
print("Predicted Price for Bike Age=4, Mileage=74092:",price_of_car[1])