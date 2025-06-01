import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE


data = pd.read_csv("healthcare-dataset-stroke-data.csv")
data.drop('id', axis=1, inplace=True)
data['bmi'] = data['bmi'].fillna(data['bmi'].mean())  # Avoid chained assignment warning


le = LabelEncoder()
for col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
    data[col] = le.fit_transform(data[col])


X = data.drop('stroke', axis=1)
y = data['stroke']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


model = RandomForestClassifier(random_state=42)
model.fit(X_train_resampled, y_train_resampled)


y_pred = model.predict(X_test)


print(" Accuracy Score:", round(accuracy_score(y_test, y_pred), 4))
print("\n Classification Report:\n", classification_report(y_test, y_pred))
print("🧮 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='coolwarm')
plt.title("Confusion Matrix ")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


importances = model.feature_importances_
features = X.columns
feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df) 
plt.title("Feature Importance")  
plt.tight_layout()
plt.show()


#  CROSS VALIDATION
cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=5)
print(f"Mean Cross-Validation Accuracy: {cv_scores.mean():.4f}")
