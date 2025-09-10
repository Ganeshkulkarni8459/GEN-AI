import pandas as pd

from sklearn.impute import KNNImputer #near neighbhour

# df = pd.read_csv('salary_data.csv')

df = pd.read_csv('modifiedData.csv')

imputer = KNNImputer(n_neighbors=3)

df['Salary'] = imputer.fit_transform(df[['Salary']])

print(df)