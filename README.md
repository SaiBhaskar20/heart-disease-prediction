# Heart Disease Prediction — Educational Demo

Flask UI for the scikit-learn model saved from the notebook.

## Run locally
Use the same model file exported by your notebook, named `heart_disease_model.pkl`, in this project folder.

```bash
python -m venv venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000. Health check: http://127.0.0.1:5000/health

## Important model compatibility note
The app uses `model.feature_names_in_` to construct the prediction DataFrame in the exact feature order expected by the saved pipeline. The form contains the fields present in the training notebook, including `Exercise Induced Angina`. If the model expects a feature that the form does not supply, the app displays an error rather than silently guessing.

Install the same scikit-learn version used to export the pickle/joblib artifact. Pickle/joblib files should only be loaded if trusted.

## Disclaimer
This is an educational project, not a medical device, diagnosis, or medical advice. Dataset provenance, target labels, leakage, and external validation must be reviewed before making any real-world claims.
