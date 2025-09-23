import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score , root_mean_squared_error

df = pd.read_csv("Advertising.csv")

x = df.drop(['sales'],axis=1)
y = df['sales']

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.8, random_state=42)
model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

# print(y_pred)

# Mean Squared Error (MSE)
# Mean Squared Error (MSE) is the average of the squared differences between the predicted and actual values.
# ✅ Use MSE if you want to penalize big errors more (useful when large mistakes are costly).
# It is a measure of how close the regression line is to the actual data points.
# The smaller the MSE, the closer the fit is to the data.
# The MSE has the units squared of whatever is plotted on the vertical axis.
# value of mse is 0 means model is good
# value of mse is near to 0 means model is good
# value of mse is near to 1 means model is good
# value of mse is near to 0 means model is bad
mse = mean_squared_error(y_test,y_pred)
print(f"Mean Squared Error: {mse}")

# Mean Absolute Error (MAE)
# Mean Absolute Error (MAE) is the average of the absolute differences between the predicted and actual values.
## ✅ Use MAE if you want a simple interpretation (average error in actual units).
# It is a measure of how close the regression line is to the actual data points.
# The smaller the MAE, the closer the fit is to the data.
# The MAE has the same units as the data being plotted on the vertical axis.
# value of mae is 0 means model is good
# value of mae is near to 0 means model is good
mae = mean_absolute_error(y_test,y_pred)
print(f"Mean Absolute Error: {mae}")

# Root Mean Squared Error (RMSE)
# Root Mean Squared Error (RMSE) is the square root of the average of the squared differences between the predicted and actual values.
# It is a measure of how close the regression line is to the actual data points.
# The smaller the RMSE, the closer the fit is to the data.
# The RMSE has the same units as the data being plotted on the vertical axis.
# value of rmse is 0 means model is good
# Lower the error, better the model
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

# R2 Score (Coefficient of Determination)
# R-squared is a statistical measure of how close the data are to the fitted regression line.
# It is also known as the coefficient of determination, or the coefficient of multiple determination for multiple regression.
# The definition of R-squared is fairly straight-forward; it is the percentage of the response variable variation that is explained by a linear model.
# value of r2 score is between 0 to 1
# value of r2 score is near to 1 means model is good
# value of r2 score is near to 0 means model is bad
# value of r2 score is negative means model is worst
r2 = r2_score(y_test,y_pred)
print(f"R-squared: {r2}")

sales = model.predict(pd.DataFrame({'TV':[100],'radio':[100],'newspaper':[100]}))
print(f"Sales: {sales}")
