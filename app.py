from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# Define the request model
class TextRequest(BaseModel):
    text: str

app = FastAPI()

# Load model and vectorizer
model = joblib.load('model_sentimen_analis.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.get("/")
def read_root():
    return {"message": "Sentiment Analysis API is running!"}

@app.post("/predict")
def predict(request: TextRequest):
    text = request.text
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    X = vectorizer.transform([text])
    prediction = model.predict(X)
    return {"prediction": prediction[0]}
