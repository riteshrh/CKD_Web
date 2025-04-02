from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)

CORS(app)

with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)
       
columns = [
    'specific_gravity', 'hemoglobin', 'serum_creatinine', 'albumin',
    'packed_cell_volume', 'diabetes_mellitus', 'hypertension',
    'blood_glucose_random', 'red_blood_cell_count', 'blood_urea'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        features = np.array([
            float(data['specific_gravity']),
            float(data['hemoglobin']),
            float(data['serum_creatinine']),
            float(data['albumin']),
            float(data['packed_cell_volume']),
            int(data['diabetes_mellitus']),
            int(data['hypertension']),
            float(data['blood_glucose_random']),
            float(data['red_blood_cell_count']),
            float(data['blood_urea']),
        ])

        features_df = pd.DataFrame([features], columns=columns)
        prediction = model.predict(features_df)

        prediction_result = int(prediction[0])

        print("Predictions:", prediction_result)
        
        return jsonify({'prediction': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("App is running")
    app.run(debug=True)
