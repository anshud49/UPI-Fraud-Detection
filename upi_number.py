import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
data = pd.read_csv("data/upi_fraud_dataset.csv")

# Ensure 'trans_date' column exists by combining year, month, and day
data.rename(columns={
    'trans_year': 'year',
    'trans_month': 'month',
    'trans_day': 'day'
}, inplace=True)
data['trans_date'] = pd.to_datetime(data[['year', 'month', 'day']])

# Feature Engineering: Aggregate features per 'upi_number'
aggregated_features = data.groupby('upi_number').agg(
    monthly_txn_count=('Id', lambda x: x.count() / data['trans_date'].dt.to_period('M').nunique()),
    hourly_txn_count=('Id', lambda x: x.count() / 24),  # Assuming uniform distribution over hours
    avg_txn_amount=('trans_amount', 'mean'),
    max_txn_amount=('trans_amount', 'max'),
    min_txn_amount=('trans_amount', 'min'),
    std_txn_amount=('trans_amount', 'std'),
    most_common_category=('category', lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'),
    most_common_state=('state', lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'),
    past_fraud_count=('fraud_risk', 'sum'),
    total_txn_count=('Id', 'count')
).reset_index()

# Calculate fraud_rate
aggregated_features['fraud_rate'] = aggregated_features['past_fraud_count'] / aggregated_features['total_txn_count']

# Merge aggregated features back to the original data
data = data.merge(aggregated_features, on='upi_number', how='left')

# Select features for modeling
feature_columns = [
    'monthly_txn_count',
    'hourly_txn_count',
    'avg_txn_amount',
    'max_txn_amount',
    'min_txn_amount',
    'std_txn_amount',
    'fraud_rate'
]

# Handle missing values if any
data[feature_columns] = data[feature_columns].fillna(0)

# Define target variable
X = data[feature_columns]
y = data['fraud_risk']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Enhanced Model Accuracy: {accuracy:.2f}")

# Save the model and feature columns
joblib.dump((rf_model, feature_columns), "FinalModels/Upi_number.pkl")
print("Enhanced model saved as Upi_number.pkl")
