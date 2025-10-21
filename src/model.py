# src/model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

def build_model(random_state=42):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=50, random_state=random_state))
    ])
    return pipeline

def save_model(model, path):
    joblib.dump(model, path)
