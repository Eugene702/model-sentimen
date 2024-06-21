import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Determine absolute path to models directory
models_dir = os.path.join(os.getcwd(), 'models')

# Load model and vectorizer using absolute paths
model_path = os.path.join(models_dir, 'model_sentimen_analis.pkl')
vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

class TextData(BaseModel):
    text: str

@app.post("/predict")
async def predict(data: TextData):
    try:
        # Transform the input text
        X = vectorizer.transform([data.text])
        prediction = model.predict(X)
        # Convert numpy.int64 to int
        prediction_int = int(prediction[0])
        return {"text": data.text, "prediction": prediction_int}
    except Exception as e:
        return {"error": str(e)}
