import os
import wandb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import uuid

PROJECT = os.environ.get("WANDB_PROJECT", "MLOPS-CAPSTONE")
ENTITY = os.environ.get("WANDB_ENTITY", "raja-ramiz-mukhtar6-szabist")

def main():
    run = wandb.init(project=PROJECT, entity=ENTITY, job_type="train")

    # Load dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model pipeline
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("✅ Accuracy:", acc)

    # Save model
    model_filename = f"model_{uuid.uuid4().hex}.joblib"
    model_path = f"./artifacts/{model_filename}"
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, model_path)

    # Log model to W&B
    artifact = wandb.Artifact("model", type="model", metadata={"accuracy": acc, "features": feature_names.tolist()})
    artifact.add_file(model_path)
    run.log_artifact(artifact, aliases=["latest"])

    run.finish()
    print(f"✅ Training done. Model saved: {model_path}")

if __name__ == "__main__":
    main()
