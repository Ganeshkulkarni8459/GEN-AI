import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
df = pd.read_csv('diabetes.csv')

# Define independent and dependent variables
X = df.drop('Outcome', axis=1)
Y = df['Outcome']

# Split the data (although not strictly necessary for saving, it's good practice)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=231231123)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Save the trained model to a file
# The model will be saved as 'diabetes_model.joblib'
joblib.dump(model, 'diabetes_model.joblib')

print("Model saved successfully as 'diabetes_model.joblib'")