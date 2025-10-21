import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import wandb
import pandas as pd
import glob
import shutil
from fastapi.middleware.cors import CORSMiddleware

# ----------------------------
# FastAPI app setup
# ----------------------------
app = FastAPI(title="MLOps Capstone Inference Service")

# Enable CORS (so Streamlit on port 8501 can access backend on 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Config and paths
# ----------------------------
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "MLOPS-CAPSTONE")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "raja-ramiz-mukhtar6-szabist")
MODEL_LOCAL_PATH = "model.joblib"  # Use local path for easy dev on Windows
MODEL_INFO = {}

# ----------------------------
# Download latest model from W&B
# ----------------------------
def download_latest_model():
    """Download latest model artifact from W&B and save locally."""
    api = wandb.Api()
    artifact_name = f"{WANDB_ENTITY}/{WANDB_PROJECT}/model:latest"
    artifact = api.artifact(artifact_name)
    artifact_dir = artifact.download(root="tmp_model")

    # Find any joblib or pkl file
    files = glob.glob(os.path.join(artifact_dir, "*.joblib")) + glob.glob(os.path.join(artifact_dir, "*.pkl"))
    if not files:
        raise FileNotFoundError("No model file found in downloaded artifact")

    src = files[0]
    shutil.copy(src, MODEL_LOCAL_PATH)
    # capture some artifact info for the API
    try:
        MODEL_INFO.update({
            "name": getattr(artifact, "name", None),
            "id": getattr(artifact, "id", None),
            "aliases": getattr(artifact, "aliases", None),
            "metadata": getattr(artifact, "metadata", {}),
        })
    except Exception:
        pass
    print(f"✅ Model downloaded from W&B to {MODEL_LOCAL_PATH}")

# ----------------------------
# Startup event
# ----------------------------
@app.on_event("startup")
def startup_event():
    try:
        download_latest_model()
    except Exception as e:
        print("⚠️ Warning: could not download model at startup:", e)

# ----------------------------
# Request / Response Models
# ----------------------------
class PredictRequest(BaseModel):
    inputs: list  # List of feature lists
    include_explanations: bool = False
    top_k: int = 5

# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    """Run inference using the trained model."""
    if not os.path.exists(MODEL_LOCAL_PATH):
        return {"error": "Model not found. Please retrain or download it."}
    try:
        model = joblib.load(MODEL_LOCAL_PATH)
        X = pd.DataFrame(req.inputs)
        preds = model.predict(X).tolist()
        # Provide human readable labels along with numeric predictions for clients
        label_map = {0: "malignant", 1: "benign"}
        labels = [label_map.get(int(p), str(p)) for p in preds]

        # If the model supports predict_proba, include class probabilities for better UX
        probabilities = None
        try:
            if hasattr(model, "predict_proba"):
                proba_arr = model.predict_proba(X)
                # assume column 0 -> class 0 (malignant), column 1 -> class 1 (benign)
                probabilities = []
                for row in proba_arr:
                    probabilities.append({"malignant": float(row[0]), "benign": float(row[1])})
        except Exception:
            # If anything goes wrong, just skip probabilities
            probabilities = None

        resp = {"predictions": preds, "labels": labels}
        if probabilities is not None:
            resp["probabilities"] = probabilities
        # Optional explanations using shap (per-row top-k features)
        if getattr(req, "include_explanations", False):
            try:
                import shap
                # if model is a pipeline, extract the tree estimator
                estimator = model
                if hasattr(model, "named_steps"):
                    # try common names
                    estimator = model.named_steps.get("rf") or model.named_steps.get("clf") or model.named_steps.get(list(model.named_steps.keys())[-1])
                explainer = shap.TreeExplainer(estimator)
                shap_vals = explainer.shap_values(X)
                # shap_values for binary classification may come as list [class0, class1]
                if isinstance(shap_vals, list) and len(shap_vals) == 2:
                    # use values for positive class (1 / benign)
                    shap_for_class = shap_vals[1]
                else:
                    shap_for_class = shap_vals

                explanations = []
                top_k = max(1, int(getattr(req, "top_k", 5)))
                for row_vals in shap_for_class:
                    # get top-k absolute contributors
                    idx = list(range(len(row_vals)))
                    abs_vals = [abs(float(v)) for v in row_vals]
                    sorted_idx = sorted(idx, key=lambda i: abs_vals[i], reverse=True)[:top_k]
                    explanations.append([{"feature": int(i), "shap_value": float(row_vals[i])} for i in sorted_idx])
                resp["explanations"] = explanations
            except Exception as e:
                # If shap isn't available or fails, skip explanations
                resp["explanations_error"] = str(e)
        return resp
    except Exception as e:
        print("❌ Prediction error:", e)
        return {"error": str(e)}


@app.get("/model_info")
def model_info():
    """Return basic model / artifact metadata for UI display."""
    info = {"model_exists": os.path.exists(MODEL_LOCAL_PATH)}
    info.update(MODEL_INFO or {})
    return info
