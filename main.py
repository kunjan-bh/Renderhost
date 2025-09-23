from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import pandas as pd
import joblib
import requests
from bs4 import BeautifulSoup
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# -----------------------------
# App setup
# -----------------------------
app = FastAPI(title="Phishing Detection API")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# -----------------------------
# Input schema
# -----------------------------
class URLInput(BaseModel):
    url: str

# -----------------------------
# Load model
# -----------------------------
MODEL_PATH = "model.pkl"  # ensure this is in your repo
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully from repo")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# -----------------------------
# Feature extraction
# -----------------------------
def extract_features_from_url(url: str) -> pd.DataFrame:
    features = {}
    try:
        response = requests.get(url, timeout=5)
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
    except Exception:
        soup = None

    features['has_title'] = int(bool(soup and soup.title and soup.title.string))
    features['has_input'] = int(bool(soup.find_all('input')) if soup else 0)
    features['has_button'] = int(bool(soup.find_all('button')) if soup else 0)
    features['has_image'] = int(bool(soup.find_all('img')) if soup else 0)
    features['has_submit'] = int(bool(soup.find_all('input', {'type':'submit'})) if soup else 0)
    features['has_link'] = int(bool(soup.find_all('a')) if soup else 0)
    features['has_password'] = int(bool(soup.find_all('input', {'type':'password'})) if soup else 0)
    features['has_email_input'] = int(bool(soup.find_all('input', {'type':'email'})) if soup else 0)
    features['has_hidden_element'] = int(bool(soup.find_all('input', {'type':'hidden'})) if soup else 0)
    features['has_audio'] = int(bool(soup.find_all('audio')) if soup else 0)
    features['has_video'] = int(bool(soup.find_all('video')) if soup else 0)
    features['number_of_inputs'] = len(soup.find_all('input')) if soup else 0
    features['number_of_buttons'] = len(soup.find_all('button')) if soup else 0
    features['number_of_images'] = len(soup.find_all('img')) if soup else 0
    features['number_of_option'] = len(soup.find_all('option')) if soup else 0
    features['number_of_list'] = len(soup.find_all('ul')) + len(soup.find_all('ol')) if soup else 0
    features['number_of_th'] = len(soup.find_all('th')) if soup else 0
    features['number_of_tr'] = len(soup.find_all('tr')) if soup else 0
    features['number_of_href'] = len(soup.find_all('a')) if soup else 0
    features['number_of_paragraph'] = len(soup.find_all('p')) if soup else 0
    features['number_of_script'] = len(soup.find_all('script')) if soup else 0
    features['length_of_title'] = len(soup.title.string) if soup and soup.title and soup.title.string else 0
    features['has_h1'] = int(bool(soup.find_all('h1')) if soup else 0)
    features['has_h2'] = int(bool(soup.find_all('h2')) if soup else 0)
    features['has_h3'] = int(bool(soup.find_all('h3')) if soup else 0)
    features['length_of_text'] = len(soup.get_text()) if soup else 0
    features['number_of_clickable_button'] = len(soup.find_all('button')) if soup else 0
    features['number_of_a'] = len(soup.find_all('a')) if soup else 0
    features['number_of_img'] = len(soup.find_all('img')) if soup else 0
    features['number_of_div'] = len(soup.find_all('div')) if soup else 0
    features['number_of_figure'] = len(soup.find_all('figure')) if soup else 0
    features['has_footer'] = int(bool(soup.find_all('footer')) if soup else 0)
    features['has_form'] = int(bool(soup.find_all('form')) if soup else 0)
    features['has_text_area'] = int(bool(soup.find_all('textarea')) if soup else 0)
    features['has_iframe'] = int(bool(soup.find_all('iframe')) if soup else 0)
    features['has_text_input'] = int(bool(soup.find_all('input', {'type':'text'})) if soup else 0)
    features['number_of_meta'] = len(soup.find_all('meta')) if soup else 0
    features['has_nav'] = int(bool(soup.find_all('nav')) if soup else 0)
    features['has_object'] = int(bool(soup.find_all('object')) if soup else 0)
    features['has_picture'] = int(bool(soup.find_all('picture')) if soup else 0)
    features['number_of_sources'] = len(soup.find_all('source')) if soup else 0
    features['number_of_span'] = len(soup.find_all('span')) if soup else 0
    features['number_of_table'] = len(soup.find_all('table')) if soup else 0
    features['URL'] = url

    column_order = list(features.keys())
    df_features = pd.DataFrame([features])[column_order]
    return df_features

# -----------------------------
# Health check / index page
# -----------------------------
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(input_data: URLInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    url = input_data.url
    df_features = extract_features_from_url(url)
    df_features_model = df_features.drop(columns=['URL'])
    prediction = model.predict(df_features_model)

    return {
        "url": url,
        "prediction": int(prediction[0])
    }
