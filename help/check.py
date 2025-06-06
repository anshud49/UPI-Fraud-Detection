# # train_model.py

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# import joblib

# # Load the dataset
# data = pd.read_csv("upi_fraud_dataset.csv")

# # Split the dataset into features and target
# X = data.drop(['Id', 'upi_number', 'fraud_risk'], axis=1)
# y = data['fraud_risk']

# # Split the dataset into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Train Random Forest model
# rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)

# # Train Logistic Regression model
# lr_model = LogisticRegression(max_iter=1000, random_state=42)
# lr_model.fit(X_train, y_train)

# # Save both models
# joblib.dump(rf_model, "rf_model.pkl")
# joblib.dump(lr_model, "lr_model.pkl")

# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib


data = pd.read_csv("../data/upi_fraud_dataset.csv")


features = ['trans_hour', 'trans_day', 'trans_month', 'trans_year', 'upi_number', 'age', 'trans_amount', 'state', 'zip']

X = data[features]
y = data['fraud_risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


joblib.dump((rf_model, features), "../FinalModels/rf_model.pkl")

print("Model trained and saved successfully with feature order.")