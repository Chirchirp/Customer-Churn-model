from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the saved Random Forest model
model = joblib.load('churn_model.pkl')

@app.route('/')
def home():
    return "Customer Churn Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request (assumes JSON input)
    data = request.get_json()

    # Convert data to DataFrame (ensure the input matches the training data)
    df = pd.DataFrame(data, index=[0])

    # Make a prediction
    prediction = model.predict(df)

    # Return the prediction as a JSON response
    return jsonify({'prediction': int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)

