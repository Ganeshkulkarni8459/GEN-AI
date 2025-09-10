import pandas as pd
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

data_frame = pd.read_csv('Advertising.csv')

x = data_frame.drop(['sales','newspaper'], axis=1)
y = data_frame['sales']

model = LinearRegression()

model.fit(x, y)

sales_predictions = model.predict(pd.DataFrame([[230.1, 37.8], [44.5, 39.3]], columns=['TV', 'radio']))

print("Predicted sales for TV=230.1, Radio=37.8", sales_predictions[0])
print("Predicted sales for TV=44.5, Radio=39.3", sales_predictions[1])

correlations_tv_sales = data_frame['TV'].corr(data_frame['sales'])
print(f'Correlation between TV and Sales: {correlations_tv_sales}')

correlations_radio_sales = data_frame['radio'].corr(data_frame['sales'])
print(f'Correlation between Radio and Sales: {correlations_radio_sales}')
