import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import BayesianRidge
from sklearn.impute import IterativeImputer

df = pd.read_csv('salary_data.csv')

imputer = IterativeImputer(estimator=BayesianRidge(),max_iter=5,random_state=0)

df['Salary'] = imputer.fit_transform(df[['Salary']])

print(df)