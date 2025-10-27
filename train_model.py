import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import joblib
import os

# -------------------------------
# 1️⃣ Setup paths
# -------------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# -------------------------------
# 2️⃣ Load dataset
# -------------------------------
print("📂 Loading dataset...")
data = pd.read_csv(r"C:\Users\janap\Downloads\archive (5)\creditcard.csv")
print(f"✅ Data loaded successfully with shape: {data.shape}")

# -------------------------------
# 3️⃣ Separate features and labels
# -------------------------------
X = data.drop("Class", axis=1)
y = data["Class"]

# -------------------------------
# 4️⃣ Scale features
# -------------------------------
print("⚙️ Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 5️⃣ Detect anomalies using Isolation Forest
# (LOF skipped because too slow for full dataset)
# -------------------------------
print("🔍 Running Isolation Forest (fast outlier detection)...")
iso = IsolationForest(contamination=0.01, random_state=42)
iso_preds = iso.fit_predict(X_scaled)
mask = iso_preds != -1  # keep non-outliers
X_clean, y_clean = X_scaled[mask], y[mask]
print(f"✅ Outlier removal done. Remaining samples: {X_clean.shape[0]}")

# -------------------------------
# 6️⃣ Handle class imbalance using SMOTE
# -------------------------------
print("⚖️ Applying SMOTE to balance classes...")
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_clean, y_clean)
print(f"✅ Balanced dataset shape: {X_res.shape}")

# -------------------------------
# 7️⃣ Split train-test
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)
print("📊 Data split done (80/20).")

# -------------------------------
# 8️⃣ Train XGBoost Classifier
# -------------------------------
print("🚀 Training XGBoost model...")
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)
model.fit(X_train, y_train)
print("✅ Model training completed.")

# -------------------------------
# 9️⃣ Evaluate model
# -------------------------------
print("📈 Evaluating model performance...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

# -------------------------------
# 🔟 Plot ROC curve
# -------------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"XGBoost (AUC = {roc_auc_score(y_test, y_prob):.3f})", linewidth=2)
plt.plot([0,1], [0,1], '--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Fraud Detection")
plt.legend()
plt.tight_layout()
plt.savefig("results/roc_curve.png")
plt.show()
print("✅ ROC Curve saved to results/roc_curve.png")

# -------------------------------
# 11️⃣ Save model and scaler
# -------------------------------
joblib.dump(model, "models/fraud_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("✅ Model and scaler saved successfully in 'models' folder!")
