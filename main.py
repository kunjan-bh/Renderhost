from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import requests
from bs4 import BeautifulSoup
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
import mlflow
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request
import redis
import hashlib
import pickle
from sqlalchemy import create_engine, inspect, text


db_url = "mysql+pymysql://kunjan:ohmy24god@127.0.0.1:3308/phishing"
table_name = "User_input_bigtable"
# Connect to DB
engine = create_engine(db_url)
inspector = inspect(engine)
table_exists = table_name in inspector.get_table_names()

# Redis client setup
redis_client = redis.Redis(host="localhost", port=6379, db=0)

# ---- MLflow Config ----
MLFLOW_TRACKING_URI = "http://localhost:5000"  # your MLflow server
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

LOCAL_MODEL_PATH = "/home/kunjan/MlopsFinal/Model/BackupLocalModel"
model = None 
model_status_message = "" 

# Use alias instead of version number
MODEL_URI = "models:/PhishingDetectionModel@staging"   # or @production




app = FastAPI(title="Phishing Detection API")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve templates (HTML)
templates = Jinja2Templates(directory="templates")
# -----------------------------
# Input schema
# -----------------------------
class URLInput(BaseModel):
    url: str



def load_model_with_fallback():
    global model
    try:
        print("📌 Trying to load model from MLflow...")
        model = mlflow.sklearn.load_model(MODEL_URI)
        print("✅ Model loaded from MLflow.")

        # Always refresh the local copy
        if os.path.exists(LOCAL_MODEL_PATH):
            import shutil
            shutil.rmtree(LOCAL_MODEL_PATH)  # remove old version
        mlflow.sklearn.save_model(model, LOCAL_MODEL_PATH)
        print(f"💾 Latest model cached at {LOCAL_MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ Could not load from MLflow: {e}")
        if os.path.exists(LOCAL_MODEL_PATH):
            print("📂 Loading fallback model from local cache...")
            model = mlflow.sklearn.load_model(LOCAL_MODEL_PATH)
            print("✅ Local cached model loaded.")
        else:
            print("❌ No cached model found. FastAPI cannot make predictions.")
            model = None
            
load_model_with_fallback()          

def extract_features_from_url(url: str) -> pd.DataFrame:
    """
    Extracts structured content-based features from a given URL.
    Returns a DataFrame with one row and columns in the same order as model training.
    """
    features = {}

    try:
        response = requests.get(url, timeout=5)
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
    except Exception as e:
        # If the page cannot be fetched, fill features with 0
        soup = None

    # -------------------- Basic content features --------------------
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

    # Convert to DataFrame with **columns in the same order** as your model expects
    column_order = [
        'has_title','has_input','has_button','has_image','has_submit','has_link','has_password',
        'has_email_input','has_hidden_element','has_audio','has_video','number_of_inputs',
        'number_of_buttons','number_of_images','number_of_option','number_of_list','number_of_th',
        'number_of_tr','number_of_href','number_of_paragraph','number_of_script','length_of_title',
        'has_h1','has_h2','has_h3','length_of_text','number_of_clickable_button','number_of_a',
        'number_of_img','number_of_div','number_of_figure','has_footer','has_form','has_text_area',
        'has_iframe','has_text_input','number_of_meta','has_nav','has_object','has_picture',
        'number_of_sources','number_of_span','number_of_table','URL'
    ]
    
    df_features = pd.DataFrame([features])[column_order]
    return df_features


# -----------------------------
# Health check
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

    # Use SHA256 hash of URL as Redis key
    url_hash = hashlib.sha256(url.encode()).hexdigest()

    # Check Redis for cached prediction
    cached_prediction = redis_client.get(url_hash)
    if cached_prediction is not None:
        prediction = pickle.loads(cached_prediction)
        return {
            "url": url,
            "prediction": int(prediction),
            "cached": True  # Indicate that result came from Redis
        }

    # Extract features and predict
    df_features = extract_features_from_url(url)
    df_features_model = df_features.drop(columns=['URL'])
    prediction = model.predict(df_features_model)

    
    
    db_inserted = False

    try:
        result = 0
        if table_exists:
            with engine.connect() as conn:
                result = conn.execute(
                    text(f"SELECT COUNT(*) FROM {table_name} WHERE URL = :url"),
                    {"url": url}
                ).scalar()
        
        if table_exists and result == 0:
            df_features['label'] = prediction[0]
            with engine.begin() as conn:  # safer: auto-commit / rollback
                df_features.to_sql(
                    table_name,
                    conn,
                    if_exists="append",
                    index=False
                )
            db_inserted = True
            print("✅ Data inserted into MySQL!")
    except Exception as e:
        print(f"⚠️ MySQL not available or insert failed, skipping DB. Error: {e}")

        
    if db_inserted==False:
        print("Data not inserted!")
    # Cache prediction in Redis
    redis_client.setex(url_hash, 3600, pickle.dumps(prediction[0]))  # Expires in 1 hour
    
    return {
        "url": url,
        "prediction": int(prediction[0]),
        "cached": False,
        "db_inserted": db_inserted
    }




