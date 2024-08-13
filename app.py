from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('best_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['features']
    features = np.array(data.split(',')).reshape(1, -1).astype(float)
    prediction = model.predict(features)
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)