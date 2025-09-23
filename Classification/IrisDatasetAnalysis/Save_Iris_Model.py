import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
df = pd.read_csv('iris.csv')

# Define independent and dependent variables
x = df.drop('species', axis=1)
y = df['species']

# Encode the species names
encoder = LabelEncoder()
species_encoded = encoder.fit_transform(y)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, species_encoded, test_size=0.2, random_state=231231123)

# Train the model
model = LogisticRegression(max_iter=2000)
model.fit(x_train, y_train)

# Save the trained model and the encoder to files
joblib.dump(model, 'iris_model.joblib')
joblib.dump(encoder, 'iris_encoder.joblib')

print("Iris model and encoder saved successfully.")