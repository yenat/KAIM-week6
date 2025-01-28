from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained models
log_reg_model = joblib.load('/home/enat/KAIM-week6/notebooks/log_reg_model.pkl')
rf_model = joblib.load('/home/enat/KAIM-week6/notebooks/rf_model.pkl')

# Encoding function for categorical features
def encode_features(data):
    # Example: encoding CurrencyCode, CountryCode, ProductCategory, and ChannelId
    encoding_dict = {
        'CurrencyCode': {'USD': 1, 'EUR': 2, 'GBP': 3},  # Add all relevant encodings
        'CountryCode': {'US': 1, 'UK': 2, 'FR': 3},  # Add all relevant encodings
        'ProductCategory': {
            'airtime': 1,
            'data_bundles': 2,
            'financial_services': 3,
            'movies': 4,
            'other': 5,
            'ticket': 6,
            'transport': 7,
            'tv': 8,
            'utility_bill': 9
        },
        'ChannelId': {1: 1, 2: 2, 3: 3}  # Update with actual encoding logic
    }

    # Encode each categorical feature using the encoding_dict
    for col, encoding in encoding_dict.items():
        if col in data:
            data[col] = encoding.get(data[col], 0)  # 0 for unknown categories

    return data

@app.route('/')
def home():
    return "Welcome to the Model Serving API!"

# Define the predict endpoint for Logistic Regression model
@app.route('/predict_log_reg', methods=['POST'])
def predict_log_reg():
    try:
        data = request.get_json()
        input_data = pd.DataFrame([data])
        input_data = input_data.apply(lambda x: encode_features(x), axis=1)  # Encode features
        prediction = log_reg_model.predict(input_data)
        return jsonify({'LogisticRegressionPrediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

# Define the predict endpoint for Random Forest model
@app.route('/predict_rf', methods=['POST'])
def predict_rf():
    try:
        data = request.get_json()
        input_data = pd.DataFrame([data])
        input_data = input_data.apply(lambda x: encode_features(x), axis=1)  # Encode features
        prediction = rf_model.predict(input_data)
        return jsonify({'RandomForestPrediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5003)
