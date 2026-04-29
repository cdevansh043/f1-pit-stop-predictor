import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

X_train, X_test, y_train, y_test, FEATURES = joblib.load('processed_data.pkl')

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    results[name] = {'model': model, 'y_pred': y_pred, 'y_proba': y_proba, 'auc': auc, 'report': report}
    print(f"{name} — AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))

joblib.dump(results, 'model_results.pkl')
joblib.dump(results['XGBoost']['model'], 'best_model.pkl')
print("\nBest model (XGBoost) saved.")