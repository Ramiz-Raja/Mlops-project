import streamlit as st
import requests
import pandas as pd
import io
import os

API_URL = "http://127.0.0.1:8000"  # FastAPI backend
st.set_page_config(page_title="MLOps Capstone Inference", page_icon="🤖")

# Custom styles to make headings and badges prettier
st.markdown(
    "<style>"
    "h1 {font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-weight:700;}"
    ".big-title {font-size:32px;}"
    ".subtitle {color: #555; margin-top: -10px;}"
    ".card {background: #fff; padding: 12px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.06);}"
    ".label-badge {padding:6px 10px; border-radius: 12px; font-weight:600;}"
    "</style>", unsafe_allow_html=True)

st.markdown("<div class='big-title'>🤖 MLOps Capstone — Breast Cancer Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predicts whether a tumor is malignant or benign using 30 numeric features (same order as training dataset).</div>", unsafe_allow_html=True)

st.write("---")

import os
import requests

# Sidebar: input mode
st.sidebar.header("Input options")
input_mode = st.sidebar.selectbox("Choose input mode", ["Paste / Type rows", "Upload CSV file", "Use sample rows from dataset"]) 

example = "14.1, 2.3, 1.5, 0.2, 1.0, 0.3, 1.5, 0.7, 1.2, 2.4, 14.3, 1.2, 3.1, 1.8, 2.2, 0.9, 0.3, 1.0, 2.5, 1.1, 13.5, 1.5, 2.7, 1.4, 1.8, 0.6, 1.2, 0.5, 1.1, 0.9"

input_text = ""
uploaded_file = None

# initialize sample/demo containers
sample_df = None
demo_examples = []
feature_names = None
sample_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "breast_cancer_data.csv"))

if input_mode == "Paste / Type rows":
    input_text = st.text_area("Input data (one row per line):", example, height=150)
elif input_mode == "Upload CSV file":
    uploaded_file = st.file_uploader("Upload a CSV with 30 columns (no header required)", type=["csv"])
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file, header=None)
            # Convert rows to comma separated strings for preview/edit
            input_text = "\n".join([", ".join(map(str, row)) for row in df_upload.values.tolist()])
            st.success(f"Loaded {len(df_upload)} rows from uploaded CSV")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
elif input_mode == "Use sample rows from dataset":
    # try load local CSV in repo if exists
    if os.path.exists(sample_path):
        try:
            # read with header if present so we can extract feature names and 'target'
            sample_df = pd.read_csv(sample_path)
            # show first 3 rows as sample
            sample_preview = sample_df.head(3)
            st.write("Sample rows (first 3):")
            st.dataframe(sample_preview)
            # fill input_text with the first row as default
            if st.button("Use first sample row"):
                input_text = ", ".join(map(str, sample_df.iloc[0].values.tolist()))
        except Exception as e:
            st.error(f"Could not load sample CSV: {e}")
    else:
        st.info("No local sample CSV found in project root. You can upload your own CSV or paste rows.")

if sample_df is not None:
    # Demo examples dropdown (uses sample CSV if available)
    if os.path.exists(sample_path):
        try:
            # try to extract first 5 examples and their target labels
            for i, r in sample_df.head(5).iterrows():
                # if 'target' column exists, use it; else unknown
                tgt = r.get('target', None) if 'target' in sample_df.columns else None
                label = 'benign' if tgt == 1 else ('malignant' if tgt == 0 else 'unknown')
                # drop 'target' column for input
                row_vals = r.drop(labels=['target']) if 'target' in sample_df.columns else r
                demo_examples.append((f"Example {i+1} — {label}", ", ".join(map(str, row_vals.values.tolist()))))
            # feature names from header (exclude target if present)
            feature_names = [c for c in sample_df.columns.tolist() if c != 'target']
        except Exception:
            demo_examples = []

if demo_examples:
    sel_demo = st.sidebar.selectbox("Demo examples", [d[0] for d in demo_examples])
    if st.sidebar.button("Use selected example"):
        # find selected example and set input_text
        for label, row in demo_examples:
            if label == sel_demo:
                input_text = row
                break

# fetch model info for display
model_info = None
try:
    mi_r = requests.get(f"{API_URL}/model_info", timeout=3)
    if mi_r.ok:
        model_info = mi_r.json()
except Exception:
    model_info = None

if model_info:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model info**")
    st.sidebar.write(f"Model file present: {model_info.get('model_exists')}")
    if model_info.get('name'):
        st.sidebar.write(f"Artifact name: {model_info.get('name')}")
    if model_info.get('id'):
        st.sidebar.write(f"Artifact id: {model_info.get('id')}")
    if model_info.get('aliases'):
        st.sidebar.write(f"Aliases: {model_info.get('aliases')}")

# explanation options
include_explanations = st.sidebar.checkbox("Include explanations (SHAP)")
top_k = st.sidebar.slider("Top K features in explanations", min_value=1, max_value=10, value=5)
st.write("")

if st.button("🔍 Predict"):
    # parse input rows from text area
    rows = []
    if input_text:
        try:
            rows = [list(map(float, [x.strip() for x in r.split(",")])) for r in input_text.splitlines() if r.strip()]
        except Exception as e:
            st.error(f"Could not parse input rows: {e}")
            rows = []

    if uploaded_file is not None and (not rows):
        # if CSV uploaded and rows not parsed from text, use parsed df_upload
        try:
            rows = df_upload.values.tolist()
        except Exception:
            pass

    if not rows:
        st.error("⚠️ Please enter or upload valid data (30 numeric features per row).")
    else:
        st.info("📡 Sending request to API...")
        payload = {"inputs": rows, "include_explanations": include_explanations, "top_k": top_k}
        try:
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=60)
            r.raise_for_status()
            result = r.json()

            if "predictions" in result:
                labels = result.get("labels") or [str(int(p)) for p in result["predictions"]]
                df = pd.DataFrame({"Input": rows, "Prediction": result["predictions"], "Label": labels})

                # Prepare display table
                df_display = df.copy()
                df_display["Input_preview"] = df_display["Input"].apply(lambda r: ", ".join([str(round(x,2)) for x in r]))
                df_display = df_display[["Input_preview", "Prediction", "Label"]].rename(columns={"Input_preview": "Input"})

                # show nice cards and a downloadable CSV
                st.success("✅ Predictions computed")
                st.write("")

                # Styled table
                def color_label(val):
                    if val == "malignant":
                        return "background-color: #fff0f0; color: #9b1313; font-weight: 700; border-radius: 8px;"
                    if val == "benign":
                        return "background-color: #f0fff2; color: #006b2c; font-weight: 700; border-radius: 8px;"
                    return ""

                styled = df_display.style.applymap(color_label, subset=["Label"]) 
                st.dataframe(styled, use_container_width=True)

                # show prediction cards with confidence if available
                probs = result.get("probabilities")
                st.markdown("---")
                st.markdown("### Prediction details")
                for i, row in df.iterrows():
                    label = row["Label"]
                    pred = int(row["Prediction"])
                    confidence_pct = "N/A"
                    if probs and i < len(probs):
                        p = probs[i]
                        confidence = p.get(label, None)
                        if confidence is None:
                            confidence = p.get("benign") if pred == 1 else p.get("malignant")
                        confidence_pct = f"{(confidence*100):.1f}%" if confidence is not None else "N/A"

                    # show card
                    if label == "malignant":
                        st.markdown(f"<div class='card'><span class='label-badge' style='background:#ffd6d6;color:#900;'>MALIGNANT</span>  " \
                                    f"<strong>Confidence:</strong> {confidence_pct} — This result suggests a higher likelihood of cancer.</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='card'><span class='label-badge' style='background:#e6ffec;color:#046; padding:6px 10px;'>Benign</span>  " \
                                    f"<strong>Confidence:</strong> {confidence_pct} — This result suggests a lower likelihood of cancer.</div>", unsafe_allow_html=True)

                # if explanations available, render them below
                explanations = result.get("explanations")
                if explanations:
                    st.markdown("---")
                    st.markdown("### Explanations (top features)")
                    for i, ex in enumerate(explanations):
                        st.write(f"Row #{i+1}:")
                        # ex is a list of {feature: idx, shap_value: val}
                        pretty = []
                        for item in ex:
                            idx = item.get("feature")
                            fname = feature_names[idx] if feature_names and idx < len(feature_names) else f"feature_{idx}"
                            pretty.append(f"{fname}: {item.get('shap_value'):.4f}")
                        st.write(", ".join(pretty))

                # allow CSV download of results
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download results as CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
            else:
                st.error(f"❌ API Error: {result.get('error')}")
        except Exception as e:
            st.error(f"🚨 Request failed: {e}")
