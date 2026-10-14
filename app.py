from pathlib import Path
import logging
import os

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "heart_disease_model.pkl"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    model = joblib.load(MODEL_PATH)
    MODEL_FEATURES = list(getattr(model, "feature_names_in_", []))
    if not MODEL_FEATURES:
        raise ValueError(
            "Model has no feature_names_in_. Export a fitted scikit-learn "
            "pipeline trained with a pandas DataFrame."
        )
except Exception:
    logger.exception("Model failed to load.")
    model = None
    MODEL_FEATURES = []

FIELD_TO_COLUMN = {
    "age": "Age", "gender": "Gender", "cholesterol": "Cholesterol",
    "bp": "Blood Pressure", "heart_rate": "Heart Rate", "smoking": "Smoking",
    "alcohol": "Alcohol Intake", "exercise_hours": "Exercise Hours",
    "family_history": "Family History", "diabetes": "Diabetes",
    "obesity": "Obesity", "stress": "Stress Level", "blood_sugar": "Blood Sugar",
    "exercise_angina": "Exercise Induced Angina", "chest_pain": "Chest Pain Type",
}
NUMERIC_RULES = {
    "age": ("Age", 1, 120), "gender": ("Gender", 0, 1),
    "cholesterol": ("Cholesterol", 0, 1000), "bp": ("Blood Pressure", 0, 400),
    "heart_rate": ("Heart Rate", 0, 300), "exercise_hours": ("Exercise Hours", 0, 168),
    "family_history": ("Family History", 0, 1), "diabetes": ("Diabetes", 0, 1),
    "obesity": ("Obesity", 0, 1), "stress": ("Stress Level", 0, 10),
    "blood_sugar": ("Blood Sugar", 0, 1000),
}
CATEGORICAL_RULES = {
    "smoking": {"Never", "Former", "Current"},
    "alcohol": {"No", "Moderate", "Heavy"},
    "exercise_angina": {"No", "Yes"},
    "chest_pain": {"Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"},
}

TEXT = {
    "en": {
        "page_title": "Heart Health Assessment",
        "subtitle": "Enter the information below to view the output from the saved model.",
        "demo": "Machine Learning · Educational Demo",
        "notice_title": "Important:",
        "notice": "This educational demo is not a medical device and cannot diagnose or rule out heart disease. Do not use its output to make treatment decisions. For urgent symptoms such as chest pain or difficulty breathing, seek emergency medical care.",
        "form_title": "Patient information",
        "form_subtitle": "Complete all fields using the dataset categories.",
        "language": "Language / భాష",
        "age": "Age (years)", "gender": "Gender (dataset encoding)", "female": "Female (0)", "male": "Male (1)",
        "cholesterol": "Cholesterol", "bp": "Blood pressure value", "heart_rate": "Heart rate",
        "smoking": "Smoking", "alcohol": "Alcohol intake", "exercise_hours": "Exercise hours per week",
        "family_history": "Family history", "diabetes": "Diabetes", "obesity": "Obesity",
        "stress": "Stress level (0–10)", "blood_sugar": "Blood sugar",
        "exercise_angina": "Exercise-induced angina", "chest_pain": "Chest pain type",
        "choose": "Choose", "never": "Never", "former": "Former", "current": "Current",
        "no": "No", "moderate": "Moderate", "heavy": "Heavy", "yes": "Yes",
        "typical": "Typical Angina", "atypical": "Atypical Angina", "non_anginal": "Non-anginal Pain",
        "asymptomatic": "Asymptomatic", "run": "Run model assessment",
        "review": "Please review:", "model_error": "The model could not be loaded. Check the model file and package versions.",
        "missing_error": "The form does not supply every feature required by the saved model: ",
        "missing_tail": ". Check the model feature names and update the form.",
        "prediction_error": "Prediction failed. Check that the saved model and scikit-learn version match.",
        "required": "is required.", "whole": "must be a whole number.", "between": "must be between",
        "invalid_choice": "Please choose a valid option for",
        "result_title": "Assessment result", "result_subtitle": "Output from the submitted information.",
        "model_output": "Model output", "predicted_class": "Predicted class:",
        "positive": "The model detected patterns that may be associated with heart disease",
        "negative": "The model did not detect a positive pattern associated with heart disease",
        "model_score": "Model score",
        "score_note": "This is the model's highest class probability, not a clinically validated estimate of personal heart-disease risk.",
        "indicators": "Entered indicators (simple rules)",
        "no_indicators": "No indicators in this app's simple rule list were triggered. That does not mean there are no risk factors.",
        "next_steps": "General next steps",
        "recommend_positive_1": "The model's prediction suggests possible indicators associated with heart disease; it does not confirm a diagnosis.",
        "recommend_positive_2": "Please consult a qualified doctor for proper evaluation.",
        "recommend_positive_3": "A doctor can assess your symptoms, medical history and whether further tests are needed.",
        "recommend_negative_1": "The model did not detect a positive pattern, but this does not prove that heart disease is absent.",
        "recommend_negative_2": "If you have any symptoms, concerns or doubts, please consult a qualified doctor.",
        "recommend_negative_3": "Seek urgent medical care for chest pain, difficulty breathing, fainting, or other severe symptoms.",
        "indicator_smoking": "Current smoking selected", "indicator_family": "Family history marked yes",
        "indicator_bp": "Entered blood pressure value is 140 or higher",
        "indicator_chol": "Entered cholesterol value is 200 or higher",
        "indicator_obesity": "Obesity marked yes", "indicator_diabetes": "Diabetes marked yes",
        "indicator_stress": "Stress level entered is 7 or higher (scale 0–10)",
        "disclaimer_title": "Medical disclaimer:",
        "disclaimer": "This is an educational machine-learning demo, not a diagnosis or medical advice. A class 0 result cannot rule out disease, and a class 1 result does not confirm disease. Seek urgent care for urgent symptoms.",
        "again": "Assess another entry", "footer": "Educational project · Not for clinical use",
    },
    "te": {
        "page_title": "గుండె ఆరోగ్య అంచనా",
        "subtitle": "మోడల్ ఫలితాన్ని చూడటానికి క్రింది వివరాలను నమోదు చేయండి.",
        "demo": "మెషిన్ లెర్నింగ్ · విద్యా ప్రయోజనాల డెమో",
        "notice_title": "ముఖ్య గమనిక:",
        "notice": "ఇది విద్యా ప్రయోజనాల కోసం రూపొందించిన డెమో మాత్రమే. ఇది వైద్య పరికరం కాదు; గుండె జబ్బును నిర్ధారించదు లేదా లేదని నిర్ధారించదు. చికిత్స నిర్ణయాలకు ఈ ఫలితాన్ని ఉపయోగించవద్దు. ఛాతీ నొప్పి లేదా శ్వాస తీసుకోవడంలో ఇబ్బంది వంటి అత్యవసర లక్షణాలు ఉంటే వెంటనే అత్యవసర వైద్య సహాయం పొందండి.",
        "form_title": "వ్యక్తి వివరాలు",
        "form_subtitle": "డేటాసెట్‌లో ఉన్న ఎంపికల ప్రకారం అన్ని వివరాలను నమోదు చేయండి.",
        "language": "భాష / Language",
        "age": "వయస్సు (సంవత్సరాలు)", "gender": "లింగం (డేటాసెట్ కోడింగ్)", "female": "స్త్రీ (0)", "male": "పురుషుడు (1)",
        "cholesterol": "కొలెస్ట్రాల్", "bp": "రక్తపోటు విలువ", "heart_rate": "గుండె కొట్టుకునే వేగం",
        "smoking": "ధూమపానం", "alcohol": "మద్యపానం", "exercise_hours": "వారానికి వ్యాయామ గంటలు",
        "family_history": "కుటుంబంలో గుండె జబ్బుల చరిత్ర", "diabetes": "మధుమేహం", "obesity": "ఊబకాయం",
        "stress": "ఒత్తిడి స్థాయి (0–10)", "blood_sugar": "రక్తంలో చక్కెర",
        "exercise_angina": "వ్యాయామ సమయంలో ఛాతీ నొప్పి (ఆంజైనా)", "chest_pain": "ఛాతీ నొప్పి రకం",
        "choose": "ఎంచుకోండి", "never": "ఎప్పుడూ లేదు", "former": "గతంలో ఉండేది", "current": "ప్రస్తుతం ఉంది",
        "no": "లేదు", "moderate": "మితంగా", "heavy": "ఎక్కువగా", "yes": "అవును",
        "typical": "సాధారణ ఆంజైనా", "atypical": "అసాధారణ ఆంజైనా", "non_anginal": "ఆంజైనా కాని నొప్పి",
        "asymptomatic": "లక్షణాలు లేవు", "run": "మోడల్ అంచనా పొందండి",
        "review": "దయచేసి సరిచూడండి:", "model_error": "మోడల్‌ను లోడ్ చేయలేకపోయాం. మోడల్ ఫైల్ మరియు ప్యాకేజీ వెర్షన్‌లను తనిఖీ చేయండి.",
        "missing_error": "సేవ్ చేసిన మోడల్‌కు అవసరమైన కొన్ని వివరాలు ఫారమ్‌లో లేవు: ",
        "missing_tail": ". మోడల్ ఫీచర్ పేర్లను తనిఖీ చేసి ఫారమ్‌ను సరిచేయండి.",
        "prediction_error": "అంచనా రూపొందించడంలో సమస్య వచ్చింది. మోడల్ మరియు scikit-learn వెర్షన్‌లు సరిపోతున్నాయో చూడండి.",
        "required": "తప్పనిసరిగా నమోదు చేయాలి.", "whole": "పూర్ణ సంఖ్యగా ఉండాలి.", "between": "ఈ పరిధిలో ఉండాలి:",
        "invalid_choice": "దయచేసి సరైన ఎంపికను ఎంచుకోండి:",
        "result_title": "అంచనా ఫలితం", "result_subtitle": "మీరు నమోదు చేసిన వివరాల ఆధారంగా మోడల్ ఇచ్చిన ఫలితం.",
        "model_output": "మోడల్ ఫలితం", "predicted_class": "అంచనా తరగతి:",
        "positive": "మోడల్ అంచనా ప్రకారం గుండె జబ్బుకు సంబంధించిన సూచనలు ఉండవచ్చు",
        "negative": "మోడల్ అంచనా ప్రకారం గుండె జబ్బుకు సంబంధించిన positive సూచనలు గుర్తించబడలేదు",
        "model_score": "మోడల్ స్కోర్",
        "score_note": "ఇది మోడల్ ఇచ్చిన అత్యధిక తరగతి probability మాత్రమే; వ్యక్తిగత గుండె జబ్బు ప్రమాదానికి వైద్యపరంగా నిర్ధారించిన శాతం కాదు.",
        "indicators": "నమోదు చేసిన వివరాల్లోని సూచనలు (సాధారణ నియమాలు)",
        "no_indicators": "ఈ యాప్‌లోని సాధారణ నియమాల ప్రకారం ఎలాంటి సూచనలు గుర్తించబడలేదు. అంటే ప్రమాద కారకాలు లేవని కాదు.",
        "next_steps": "తదుపరి సాధారణ సూచనలు",
        "recommend_positive_1": "మోడల్ అంచనా ప్రకారం గుండె జబ్బుకు సంబంధించిన సూచనలు ఉండవచ్చు; ఇది వైద్య నిర్ధారణ కాదు.",
        "recommend_positive_2": "దయచేసి సరైన వైద్య పరీక్షల కోసం వైద్యుడిని సంప్రదించండి.",
        "recommend_positive_3": "మీ లక్షణాలు, వైద్య చరిత్రను పరిశీలించి మరిన్ని పరీక్షలు అవసరమా అనేది వైద్యుడు నిర్ణయిస్తారు.",
        "recommend_negative_1": "మోడల్‌లో positive సూచనలు గుర్తించబడలేదు; అయినప్పటికీ గుండె జబ్బు లేదని దీని ద్వారా నిర్ధారించలేము.",
        "recommend_negative_2": "మీకు ఏవైనా లక్షణాలు, అనుమానాలు లేదా ఆందోళనలు ఉంటే దయచేసి వైద్యుడిని సంప్రదించండి.",
        "recommend_negative_3": "ఛాతీ నొప్పి, శ్వాస తీసుకోవడంలో ఇబ్బంది, మూర్ఛ లేదా తీవ్రమైన లక్షణాలు ఉంటే వెంటనే అత్యవసర వైద్య సహాయం పొందండి.",
        "indicator_smoking": "ప్రస్తుతం ధూమపానం చేస్తున్నట్లు ఎంపిక చేశారు",
        "indicator_family": "కుటుంబ చరిత్ర ఉందని నమోదు చేశారు",
        "indicator_bp": "నమోదు చేసిన రక్తపోటు విలువ 140 లేదా అంతకంటే ఎక్కువ",
        "indicator_chol": "నమోదు చేసిన కొలెస్ట్రాల్ విలువ 200 లేదా అంతకంటే ఎక్కువ",
        "indicator_obesity": "ఊబకాయం ఉందని నమోదు చేశారు", "indicator_diabetes": "మధుమేహం ఉందని నమోదు చేశారు",
        "indicator_stress": "ఒత్తిడి స్థాయి 7 లేదా అంతకంటే ఎక్కువగా నమోదు చేశారు (0–10 స్కేల్)",
        "disclaimer_title": "వైద్యపరమైన నిరాకరణ:",
        "disclaimer": "ఇది విద్యా ప్రయోజనాల మెషిన్ లెర్నింగ్ డెమో మాత్రమే; వైద్య నిర్ధారణ లేదా వైద్య సలహా కాదు. Class 0 వచ్చినా గుండె జబ్బు లేదని చెప్పలేం; Class 1 వచ్చినా గుండె జబ్బు ఉందని నిర్ధారించలేం. అత్యవసర లక్షణాలు ఉంటే వెంటనే వైద్య సహాయం పొందండి.",
        "again": "మరో వివరాల నమోదు చేయండి", "footer": "విద్యా ప్రాజెక్ట్ · వైద్య వినియోగం కోసం కాదు",
    }
}

def get_lang(value):
    return "te" if value == "te" else "en"

def parse_form(form, lang):
    t = TEXT[lang]
    values, errors = {}, []
    for key, (column, minimum, maximum) in NUMERIC_RULES.items():
        raw = form.get(key, "").strip()
        if not raw:
            errors.append(f"{column} {t['required']}")
            continue
        try:
            value = int(raw)
        except ValueError:
            errors.append(f"{column} {t['whole']}")
            continue
        if not minimum <= value <= maximum:
            errors.append(f"{column} {t['between']} {minimum}–{maximum}.")
            continue
        values[column] = value
    for key, allowed in CATEGORICAL_RULES.items():
        value = form.get(key, "").strip()
        if value not in allowed:
            errors.append(f"{t['invalid_choice']} {FIELD_TO_COLUMN[key]}.")
        else:
            values[FIELD_TO_COLUMN[key]] = value
    return values, errors

@app.get("/")
def home():
    lang = get_lang(request.args.get("lang", "en"))
    return render_template("index.html", form_data={}, lang=lang, t=TEXT[lang])

@app.post("/submit")
def predict():
    lang = get_lang(request.form.get("lang", "en"))
    t = TEXT[lang]
    values, errors = parse_form(request.form, lang)
    if errors:
        return render_template("index.html", errors=errors, form_data=request.form, lang=lang, t=t), 400
    if model is None:
        return render_template("index.html", errors=[t["model_error"]], form_data=request.form, lang=lang, t=t), 500
    missing = [column for column in MODEL_FEATURES if column not in values]
    if missing:
        error = t["missing_error"] + ", ".join(missing) + t["missing_tail"]
        return render_template("index.html", errors=[error], form_data=request.form, lang=lang, t=t), 500
    try:
        features = pd.DataFrame([[values[column] for column in MODEL_FEATURES]], columns=MODEL_FEATURES)
        prediction = int(model.predict(features)[0])
        model_score = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            model_score = round(float(max(probabilities)) * 100, 1)

        indicators = []
        if values.get("Smoking") == "Current": indicators.append(t["indicator_smoking"])
        if values.get("Family History") == 1: indicators.append(t["indicator_family"])
        if values.get("Blood Pressure", 0) >= 140: indicators.append(t["indicator_bp"])
        if values.get("Cholesterol", 0) >= 200: indicators.append(t["indicator_chol"])
        if values.get("Obesity") == 1: indicators.append(t["indicator_obesity"])
        if values.get("Diabetes") == 1: indicators.append(t["indicator_diabetes"])
        if values.get("Stress Level", 0) >= 7: indicators.append(t["indicator_stress"])

        if prediction == 1:
            risk_level = t["positive"]
            result_kind = "high"
            recommendations = [t["recommend_positive_1"], t["recommend_positive_2"], t["recommend_positive_3"]]
        else:
            risk_level = t["negative"]
            result_kind = "low"
            recommendations = [t["recommend_negative_1"], t["recommend_negative_2"], t["recommend_negative_3"]]

        return render_template(
            "result.html", risk_level=risk_level, result_kind=result_kind,
            prediction_label=str(prediction), model_score=model_score,
            indicators=indicators, recommendations=recommendations, lang=lang, t=t
        )
    except Exception:
        logger.exception("Prediction failed.")
        return render_template("index.html", errors=[t["prediction_error"]], form_data=request.form, lang=lang, t=t), 500

@app.get("/health")
def health():
    if model is None:
        return {"status": "model_unavailable"}, 503
    return {"status": "ok", "feature_count": len(MODEL_FEATURES)}, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
