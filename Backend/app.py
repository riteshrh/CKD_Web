from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS
import pandas as pd

# Import context and Azure LLM
from contextChatbot import llm, general_context

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

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_input = data.get('message')

        if not user_input:
            return jsonify({'error': 'No message provided'}), 400

        messages = [
            ("system", 
             f"""
             You are a medical assistant specialized in Chronic Kidney Disease (CKD).
             You can respond to greetings and general conversational messages naturally. 
             Only respond using the information provided below.
             {general_context}
             Do NOT use outside knowledge or assumptions. 
             If a question is not related to CKD or the answer is not in the context, reply with: "I'm only able to answer questions related to CKD based on the given context".
             """),
            ("human", user_input)
        ]

        response = llm.invoke(messages)
        return jsonify({'response': response.content})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    print("App is running")
    app.run(debug=True)
