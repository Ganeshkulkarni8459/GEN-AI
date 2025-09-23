import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

df = pd.read_csv('salary_dataset.csv')

x = df.drop("Salary",axis=1)
y = df["Salary"]

column_transformer = ColumnTransformer(
    transformers=[
        ("onehot",OneHotEncoder(sparse_output=True,drop="first"),["Title"])
    ],
    remainder="passthrough"
)

tarnsformed_values = column_transformer.fit_transform(x)

tarnsformed_features = pd.DataFrame(tarnsformed_values,columns=column_transformer.get_feature_names_out())

model = LinearRegression()
model.fit(tarnsformed_features,y)

data_to_predict = pd.DataFrame([["Project Manager",7]],columns=["Title","Experience"])

new_data_transformed = column_transformer.transform(data_to_predict)

dataframe_to_predict = pd.DataFrame(new_data_transformed,columns=column_transformer.get_feature_names_out())

y_pred = model.predict(dataframe_to_predict)
print(y_pred)   