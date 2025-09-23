import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression , LogisticRegressionCV
from sklearn.metrics import accuracy_score , confusion_matrix , precision_score , f1_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('hearing_test.csv')

# print(df.corr())


hearing_problem = df[df['test_result']==1]
no_hearing_problem = df[df['test_result']==0]

plt.scatter(hearing_problem['age'],hearing_problem['physical_score'],color="red",marker=".")
plt.scatter(no_hearing_problem['age'],no_hearing_problem['physical_score'],color="green",marker=".")
# plt.show()

X = df.drop('test_result',axis=1)
Y = df['test_result']

x_train , x_test , y_train , y_test = train_test_split(X,Y,test_size=0.2,random_state=231231123)
model = LogisticRegression()

model.fit(x_train,y_train)

output = model.predict(pd.DataFrame([[30,28]],columns=['age','physical_score']))
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