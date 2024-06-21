from fastapi import FastAPI
import joblib
from pydantic import BaseModel

app = FastAPI()

# Load model and vectorizer
model = joblib.load('model_sentimen_analis.pkl')
vectorizer = joblib.load('vectorizer.pkl')

class TextData(BaseModel):
    text: str

@app.post("/predict")
async def predict(data: TextData):
    # Preprocess the input text
    X = vectorizer.transform([data.text])
    prediction = model.predict(X)
    # Convert numpy.int64 to int
    prediction_int = int(prediction[0])
    return {"text": data.text, "prediction": prediction_int}
