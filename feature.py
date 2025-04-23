import matplotlib.pyplot as plt
import joblib
import pandas as pd

rf_model, features = joblib.load("rf_model.pkl")

feature_importance = rf_model.feature_importances_


importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)


plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance in Random Forest')
plt.show()


print("\n📊 Features used by Random Forest Model (sorted by importance):")
print(importance_df)
