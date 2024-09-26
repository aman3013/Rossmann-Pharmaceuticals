from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model = joblib.load("../models/model_25-09-2024-18-25-55.pkl")

# Define input data model
class InputData(BaseModel):
    features: list

# Define API endpoints
@app.post("/predict")
async def predict(data: InputData):
    try:
        # Convert input data to numpy array
        input_data = np.array(data.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Return prediction
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Run the API with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)