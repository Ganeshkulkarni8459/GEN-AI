from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and the encoder
try:
    model = joblib.load('iris_model.joblib')
    encoder = joblib.load('iris_encoder.joblib')
    print("Iris model and encoder loaded successfully.")
except FileNotFoundError:
    print("Error: Model or encoder file not found. Please run 'save_iris_model.py' first.")
    exit()

@app.route('/predict_iris', methods=['GET'])
def predict_iris():
    """
    API endpoint to get a prediction from the Iris model.
    The GET request should include all features as URL parameters.
    
    Example URL:
    /predict_iris?sepal_length=7&sepal_width=3.2&petal_length=4.7&petal_width=1.4
    """
    try:
        # Get the input data from the GET request URL parameters
        sepal_length = float(request.args.get('sepal_length'))
        sepal_width = float(request.args.get('sepal_width'))
        petal_length = float(request.args.get('petal_length'))
        petal_width = float(request.args.get('petal_width'))

        # Create a DataFrame from the input data
        input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                                  columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

        # Get the numerical prediction from the loaded model
        prediction_value = model.predict(input_data)[0]

        # Convert the numerical prediction back to a species name using the encoder
        prediction_label = encoder.inverse_transform([prediction_value])[0]

        # Return the prediction in a JSON response
        return jsonify({
            "prediction_value": int(prediction_value),
            "prediction_label": prediction_label,
            "message": "Iris prediction successful"
        })

    except (ValueError, TypeError) as e:
        # Handle cases where input data is missing or invalid
        return jsonify({
            "error": "Invalid input data. Please provide all 4 features as floats.",
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
