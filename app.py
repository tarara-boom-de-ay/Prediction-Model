from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained regression model
with open('regression.pkl', 'rb') as model_file:
  regression_model = pickle.load(model_file)
# Load the pre-trained model
with open('classification.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route("/")
def Index():
    return render_template('index.html')

@app.route("/index2")
def Index2():
    return render_template('regression.html')

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        data = request.get_json()

        # Convert data to a format suitable for prediction (you may need to adjust this based on your model requirements)
        input_data = np.array([float(data['long_hair']), float(data['forehead_w']), float(data['forehead_h']),
                               float(data['nose_wide']), float(data['nose_long']), float(data['lips_thin']),
                               float(data['nose_to_lip'])]).reshape(1, -1)

        # Make the prediction
        prediction = model.predict(input_data)[0]

        # Return the prediction as JSON
        return jsonify({'prediction': int(prediction)})  # Convert the prediction to int for JSON serialization
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route("/predictreg", methods=['POST'])
def predict_reg():
    try:
        # Get the input data from the request
        data = request.get_json()

        # Convert data to a format suitable for regression prediction
        input_data = np.array([float(data['gender']), float(data['age']), float(data['sleep_duration']), float(data['sleep_quality']), float(data['stress_lvl']), float(data['bmi']), float(data['sleep_disorder']), float(data['daily_steps'])]).reshape(1, -1)

        # Make the regression prediction
        prediction = regression_model.predict(input_data)[0]

        # Return the prediction as JSON
        return jsonify({'prediction': float(prediction)})  # Convert the prediction to float for JSON serialization
    except Exception as e:
        return jsonify({'error': str(e)})
    
if __name__ == "_main_":
    app.run(debug=True)