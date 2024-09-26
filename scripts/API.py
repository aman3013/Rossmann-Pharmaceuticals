from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Load the trained model (replace with your model path)
model = joblib.load("../models/model_25-09-2024-18-25-55.pkl")

# Define the input data model (adjust based on your features)
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    # Add more fields as per your model input

# Health check route
@app.get("/")
def read_root():
    return {"message": "API is working"}

# Prediction route
@app.post("/predict")
def predict(data: InputData):
    # Convert input data to a DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Preprocessing steps (if necessary)
    # Example: input_df = preprocess(input_df)

    # Make prediction
    prediction = model.predict(input_df)

    # Return prediction as JSON response
    return {"prediction": prediction.tolist()}
