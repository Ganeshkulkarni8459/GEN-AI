import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression , LogisticRegressionCV
from sklearn.metrics import accuracy_score , confusion_matrix , precision_score , f1_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('diabetes.csv')

X = df.drop('Outcome',axis=1)
Y = df['Outcome']

x_train , x_test , y_train , y_test = train_test_split(X,Y,test_size=0.2,random_state=231231123)
model = LogisticRegression()

model.fit(x_train,y_train)

output = model.predict(pd.DataFrame([[4,110,92,0,0,37.6,0.191,30]],columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']))
print("Predicted Output : ",output)

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