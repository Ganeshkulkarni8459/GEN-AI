import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression , LogisticRegressionCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score , confusion_matrix , precision_score , f1_score
from sklearn.model_selection import train_test_split


df = pd.read_csv('iris.csv')

x = df.drop('species',axis=1)
y = df['species']

encoder = LabelEncoder()

species_encoded = encoder.fit_transform(y)

x_train , x_test , y_train , y_test = train_test_split(x,species_encoded,test_size=0.2,random_state=231231123)


model = LogisticRegression()

model.fit(x_train,y_train)

test_data_point  = model.predict(pd.DataFrame([[7,3.2,4.7,1.4]],columns=['sepal_length','sepal_width','petal_length','petal_width']))

print(encoder.inverse_transform(test_data_point))

y_pred = model.predict(x_test)

confusion = confusion_matrix(y_test , y_pred)
print("Confusion Matrix: ")
print(confusion)

accuracy = accuracy_score(y_test , y_pred)
print("Accuracy Matrix:")
print(accuracy)

precision = precision_score(y_test , y_pred,average= 'weighted')
print("######### Precision : ")
print(precision)

f1score = f1_score(y_test , y_pred,average='weighted')
print("######### F1 Score : ")
print(f1score)




