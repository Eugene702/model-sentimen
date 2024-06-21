from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import skops.io as sio

# Define the request model
class TextRequest(BaseModel):
    text: str

app = FastAPI()

# Load model and vectorizer
try:
    model = sio.load('model_sentimen_analis.skops')
    vectorizer = sio.load('vectorizer.skops')
except Exception as e:
    raise RuntimeError(f"Failed to load model or vectorizer: {e}")

@app.get("/")
def read_root():
    return {"message": "Sentiment Analysis API is running!"}

@app.post("/predict")
def predict(request: TextRequest):
    text = request.text
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    try:
        X = vectorizer.transform([text])
        prediction = model.predict(X)
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")
