
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('software_data.csv')

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

missing_values = df.isnull().sum()
if missing_values.any():
    print("Missing values found - filling with median/mode")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

df = df.drop_duplicates()

target_column = 'defect'
if target_column not in df.columns:
    target_column = df.columns[-1]
    print("Using column as target:", target_column)

X = df.drop(columns=[target_column])
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

class_ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = XGBClassifier(
    scale_pos_weight=class_ratio,
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("Improvement:", accuracy - 0.66)

print(classification_report(y_test, y_pred))

plt.figure(figsize=(10, 6))
feature_importance = model.feature_importances_
features = X.columns
indices = np.argsort(feature_importance)[::-1]

plt.bar(range(len(feature_importance)), feature_importance[indices])
plt.xticks(range(len(feature_importance)), [features[i] for i in indices], rotation=45)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

print("XGBoost implementation complete!")
