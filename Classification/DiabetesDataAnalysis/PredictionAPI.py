from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load('diabetes_model.joblib')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: The model file 'diabetes_model.joblib' was not found.")
    print("Please run the 'save_model.py' script first to save the model.")
    exit()

@app.route('/predict', methods=['GET'])
def predict():
    """
    API endpoint to get a prediction from the diabetes model.
    The GET request should include all features as URL parameters.
    
    Example URL:
    /predict?Pregnancies=4&Glucose=110&BloodPressure=92&SkinThickness=0&Insulin=0&BMI=37.6&DiabetesPedigreeFunction=0.191&Age=30
    """
    try:
        # Get the input data from the GET request URL parameters
        pregnancies = float(request.args.get('Pregnancies'))
        glucose = float(request.args.get('Glucose'))
        blood_pressure = float(request.args.get('BloodPressure'))
        skin_thickness = float(request.args.get('SkinThickness'))
        insulin = float(request.args.get('Insulin'))
        bmi = float(request.args.get('BMI'))
        dpf = float(request.args.get('DiabetesPedigreeFunction'))
        age = float(request.args.get('Age'))

        # Create a DataFrame from the input data
        input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
                                  columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

        # Get the prediction from the loaded model
        prediction = model.predict(input_data)[0]

        # Convert the prediction to a human-readable string
        outcome = "Diabetic" if prediction == 1 else "Non-Diabetic"

        # Return the prediction in a JSON response
        return jsonify({
            "prediction_value": int(prediction),
            "prediction_label": outcome,
            "message": "Prediction successful"
        })

    except (ValueError, TypeError) as e:
        # Handle cases where input data is missing or invalid
        return jsonify({
            "error": "Invalid input data. Please provide all 8 features as floats.",
            "details": str(e)
        }), 400
    except Exception as e:
        # Handle any other unexpected errors
        return jsonify({
            "error": "An unexpected error occurred.",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
