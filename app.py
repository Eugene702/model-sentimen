from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Memuat model
model = joblib.load('model_sentimen_analis.pkl')

@app.post("/predict")
async def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"prediction": prediction.tolist()}

# Menjalankan aplikasi dengan Uvicorn
# Di terminal, jalankan: uvicorn main:app --reload
