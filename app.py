import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, render_template

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model (ensure the path is correct)
pickle_in = open("Winedataset_LR1.pkl", "rb")
classifier = pickle.load(pickle_in)

@app.route('/')  # Home route
def welcome():
    return render_template('index.html')  # Render the HTML page with the input form

@app.route('/predict', methods=["Get"])  # For individual predictions
def predict_note_authentication():
    """
    Expected input features:
    ['citric Acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide']
    """
    input_cols = ['citric Acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide']
    list1 = []
    
    # Collect values from the form data (GET request)
    for i in input_cols:
        val = request.args.get(i)
        if val:
            try:
                list1.append(float(val))  # Convert input to float for prediction
            except ValueError:
                return f"Error: Invalid value for {i}. Must be a numeric value.", 400
        else:
            return f"Error: Missing value for {i}", 400  # Return error if any input is missing
    
    # Make prediction
    prediction = classifier.predict([list1])  # List of feature values
    
    return f"The predicted value is: {prediction[0]}"  # Return the prediction result

@app.route('/predict_file', methods=["POST"])  # For file uploads
def predict_note_file():
    # Read uploaded CSV file
    file = request.files.get("file")
    if not file:
        return "Error: No file uploaded", 400
    
    try:
        # Assuming the uploaded file has the correct columns as per model requirements
        df_test = pd.read_csv(file)
        
        # Ensure the necessary columns are present in the file
        required_columns = ['citric Acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide']
        if not all(col in df_test.columns for col in required_columns):
            return f"Error: Missing required columns. Expected columns: {required_columns}", 400
        
        # Make predictions
        predictions = classifier.predict(df_test[required_columns])  # Select only the necessary columns for prediction
        
        return str(list(predictions))  # Return predictions as a list
    
    except Exception as e:
        return f"Error processing file: {e}", 500  # Handle unexpected errors

# Corrected block for running the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
