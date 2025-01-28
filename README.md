
# Credit Scoring System

The goal of this project is to develop a credit scoring model that classifies users into high-risk(bad) and low-risk(good) categories. The model uses various feature engineering techniques, RFMS formalism, and machine learning algorithms. The final model is served via a REST API for real-time predictions.

## Table of Contents

- [Overview](#overview)
- [Files](#files)
- [Setup](#setup)
- [Usage](#usage)
- [Model Serving](#model-serving)


## Files

- \`EDA_credit_score.ipynb\`: Exploratory Data Analysis (EDA) for understanding the dataset.
- \`feature_engineering.ipynb\`: Feature engineering and preprocessing steps.
- \`woe_binning.ipynb\`: Weight of Evidence (WoE) binning for categorical features.
- \`build_model.ipynb\`: Model building, training, and evaluation.
- \`scripts/app.py\`: Flask API for serving the trained model.

## Setup

1. Clone the repository:
   \`\`\`bash
   git clone https://github.com/yenat/KAIM-week6.git
   \`\`\`
2. Navigate to the project directory:
   \`\`\`bash
   cd KAIM-week6
   \`\`\`
3. Install the required packages:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

## Usage

1. Open the Jupyter Notebook:
   \`\`\`bash
   jupyter notebook
   \`\`\`
2. Run the notebook cells in the following order:
   - \`EDA_credit_score.ipynb\`
   - \`feature_engineering.ipynb\`
   - \`woe_binning.ipynb\`
   - \`build_model.ipynb\`

## Model Serving

The trained model is served via a Flask API. Follow these steps to run the API:

1. Navigate to the \`scripts\` directory:
   \`\`\`bash
   cd scripts
   \`\`\`
2. Run the Flask application:
   \`\`\`bash
   python app.py
   \`\`\`
3. The API will start running on \`http://127.0.0.1:5000/\`. You can send POST requests to the \`/predict_rf\` or \`/predict_log_reg\` endpoint with the required input data to get predictions.

### Example API Request

You can use \`curl\` or any HTTP client (e.g., Postman) to interact with the API:

\`\`\`bash
curl -X POST -H "Content-Type: application/json" -d '{
  "feature1": value1,
  "feature2": value2,
  ...
}' http://127.0.0.1:5000/predict_rf
\`\`\`

Replace \`feature1\`, \`feature2\`, etc., with the actual feature names and values used in your model.

### Example API Response

The API will return a JSON response with the prediction, which will be either \`good\` or \`bad\`:

\`\`\`json
{
  "prediction": "good"
}
\`\`\`

