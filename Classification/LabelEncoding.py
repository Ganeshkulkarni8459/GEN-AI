import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('iris.csv')

x = df.drop('species',axis=1)
y = df['species']


encoder = LabelEncoder()

species_encoded = encoder.fit_transform(y)

model = LogisticRegression()

model.fit(x,species_encoded)

test_data_point  = model.predict(pd.DataFrame([[7,3.2,4.7,1.4]],columns=['sepal_length','sepal_width','petal_length','petal_width']))

print(encoder.inverse_transform(test_data_point))




