# MLOps Capstone — End to End Pipeline

This repository implements an end-to-end MLOps pipeline:
- Training script with W&B experiment tracking and artifact storage
- Model registry via W&B artifact alias `latest`
- FastAPI backend that pulls `model:latest` from W&B and serves predictions
- Streamlit frontend that calls the backend
- Dockerfiles for frontend and backend
- GitHub Actions workflows to train, register and deploy

## Quick local run (dev)
1. Create Python venv and install:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r src/requirements.txt
