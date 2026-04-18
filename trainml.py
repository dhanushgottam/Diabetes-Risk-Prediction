import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
    

df = pd.read_csv(r"C:\Users\gottam dhanush\Downloads\archive (3)\diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

print("Dataset Loaded Successfully!")
print("Shape:", df.shape)
print(df.head())

y = df['Diabetes_binary']
X = df.drop('Diabetes_binary', axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train/Test split completed")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print("\nTraining Models...\n")

lr = LogisticRegression(max_iter=500)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)

# Random Forest
rf = RandomForestClassifier(n_estimators=250, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)

# XGBoost
xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    learning_rate=0.05,
    max_depth=7,
    n_estimators=250
)
xgb.fit(X_train_scaled, y_train)
xgb_pred = xgb.predict(X_test_scaled)

# SVM
svm = SVC(probability=True)
svm.fit(X_train_scaled, y_train)
svm_pred = svm.predict(X_test_scaled)

# KNN
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_scaled, y_train)
knn_pred = knn.predict(X_test_scaled)

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
nb_pred = nb.predict(X_test_scaled)
models = {
    "Logistic Regression": lr_pred,
    "Random Forest": rf_pred,
    "XGBoost": xgb_pred,
    "SVM": svm_pred,
    "KNN": knn_pred,
    "Naive Bayes": nb_pred
}

print("\n MODEL PERFORMANCE\n")
for name, preds in models.items():
    print("\nModel:", name)
    print("Accuracy:", accuracy_score(y_test, preds))
    print("F1 Score:", f1_score(y_test, preds))
    print(classification_report(y_test, preds))

cm = confusion_matrix(y_test, xgb_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


xgb_prob = xgb.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, xgb_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve - XGBoost")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.show()


# FEATURE IMPORTANCE
from xgboost import plot_importance

plt.figure(figsize=(12, 7))
plot_importance(xgb, max_num_features=15)
plt.title("Top 15 Feature Importances - XGBoost")
plt.show()


#SAVE MODEL + SCALER + MEANS FOR FLASK

# Save best model (XGBoost)
joblib.dump(xgb, "xgboost_diabetes_model.pkl")
print("\nSaved: xgboost_diabetes_model.pkl")

# Save scaler
joblib.dump(scaler, "scaler.pkl")
print("Saved: scaler.pkl")

# Save mean values for missing 11 features
feature_means = df.drop("Diabetes_binary", axis=1).mean()
joblib.dump(feature_means, "feature_means.pkl")
print("Saved: feature_means.pkl")
