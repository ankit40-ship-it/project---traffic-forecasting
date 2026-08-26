# Traffic Pattern Forecasting Using Machine Learning

## Overview
This project predicts traffic volume using historical traffic patterns and machine learning.

## Technologies
- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest Regressor
- Streamlit
- Matplotlib
- VS Code

## Features
- Data preprocessing
- Feature engineering
- Traffic forecasting
- Random Forest ML model
- MAE, RMSE and R² evaluation
- Hourly traffic analytics
- Weather-based analysis
- Interactive Streamlit dashboard

## How to Run

### 1. Open the project in VS Code

### 2. Create/activate a virtual environment (optional)
Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install libraries
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python train_model.py
```

This creates:
`model/traffic_model.pkl`

### 5. Start the dashboard
```bash
streamlit run app.py
```

The browser will open the Traffic Pattern Forecasting dashboard.

## Project Workflow
Historical Data → Cleaning → Feature Engineering → Train/Test Split → Random Forest → Evaluation → Forecast → Dashboard

## Note
The included dataset is a synthetically generated educational dataset designed to demonstrate the complete ML workflow. For a real-world deployment, replace it with verified traffic sensor/API data.
