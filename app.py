from flask import Flask, request, render_template, session, redirect, url_for
import joblib
import pandas as pd


data = pd.read_csv("data/upi_fraud_dataset.csv")
data['upi_number'] = data['upi_number'].astype(str).str.strip().str.lower()
data.rename(columns={
    'trans_year': 'year',
    'trans_month': 'month',
    'trans_day': 'day'
}, inplace=True)
data['trans_date'] = pd.to_datetime(data[['year', 'month', 'day']])

app = Flask(__name__)
app.secret_key = '23231434341'

# Load the models
full_model, full_feature_order = joblib.load("FinalModels/rf_model.pkl")          # Full model with all parameters
upi_model, upi_features = joblib.load("FinalModels/Upi_number.pkl")                          # Model using only UPI number

@app.route('/')
def home():
    result = session.pop('result', None)
    return render_template('index.html', result=result)


@app.route('/upi-risk-scan')
def upi_risk_scan():
    return render_template('Pages/upi_risk_scan.html')

@app.route('/full-data-assessment')
def full_data_assessment():
    return render_template('Pages/full_data_assessment.html')


@app.route('/predict_upi_only', methods=['POST'])
def predict_upi_only():
    upi_number = request.form.get('upi_number', '').strip().lower()

    if not upi_number:
        session['result'] = "Please enter a UPI number."
        return redirect(url_for('upi_risk_scan'))

    # Filter data for this UPI number
    upi_data = data[data['upi_number'] == upi_number].copy()

    if upi_data.empty:
        session['result'] = "No historical data available for this UPI number."
        return redirect(url_for('upi_risk_scan'))

    # Feature engineering
    upi_data['hour'] = upi_data['trans_date'].dt.hour
    upi_data['month'] = upi_data['trans_date'].dt.month
    upi_data['year'] = upi_data['trans_date'].dt.year

    features = {
        'hourly_txn_count': upi_data.groupby('hour').size().mean(),
        'monthly_txn_count': upi_data.groupby('month').size().mean(),
        'yearly_txn_count': upi_data.groupby('year').size().mean(),
        'std_txn_amount': upi_data['trans_amount'].std(),
        'avg_txn_amount': upi_data['trans_amount'].mean(),
        'min_txn_amount': upi_data['trans_amount'].min(),
        'max_txn_amount': upi_data['trans_amount'].max(),
        'fraud_rate': upi_data['fraud_risk'].mean()
    }

    feature_df = pd.DataFrame([features])
    feature_df = feature_df[upi_features]

    # Prediction
    upi_prediction = upi_model.predict(feature_df)[0]
    result = "Fraud" if upi_prediction == 1 else "Not Fraud"

    session['result'] = result
    return redirect(url_for('upi_risk_scan'))

@app.context_processor
def inject_result():
    return dict(result=session.get('result'))

@app.route('/clear-result', methods=['POST'])
def clear_result():
    session.pop('result', None)
    return '', 204

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form
    non_empty_fields = [v for v in form_data.values() if v.strip() != '']

    if len(non_empty_fields) == 0:
        session['result'] = None
        return redirect(url_for('home'))

    try:
        input_data = {
            "trans_hour": int(form_data['trans_hour']),
            "trans_day": int(form_data['trans_day']),
            "trans_month": int(form_data['trans_month']),
            "trans_year": int(form_data['trans_year']),
            "upi_number": form_data['upi_number'],
            "age": int(form_data['age']),
            "trans_amount": float(form_data['trans_amount']),
            "state": int(form_data['state']),
            "zip": int(form_data['zip'])
        }

        df = pd.DataFrame([input_data])
        df = df[full_feature_order]

        full_prediction = full_model.predict(df)[0]
        result = "Fraud" if full_prediction == 1 else "Not Fraud"

    except Exception as e:
        result = f"Error: {str(e)}"

    session['result'] = result
    return redirect(url_for('full_data_assessment'))


if __name__ == '__main__':
    app.run(debug=True)