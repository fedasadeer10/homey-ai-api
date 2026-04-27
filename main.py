from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("property_model.pkl")
le_location = joblib.load("location_encoder.pkl")
le_type = joblib.load("type_encoder.pkl")

class ApartmentInput(BaseModel):

    location: str           # Urban / Rural / Mountainous / Nature Reserve / Coastal
    property_type: str      # Apartment / Villa
    rooms: int              # 1-8
    bathrooms: int          # 1-6
    area_sqft: float        # 100-1000
    floor: int
    year_built: int
    has_pool: int           # 0 أو 1
    has_gym: int            # 0 أو 1
    has_parking: int        # 0 أو 1
    has_security: int       # 0 أو 1
    has_activity: int       # 0 أو 1

@app.post("/predict")
def predict_price(data: ApartmentInput):
    loc_encoded = le_location.transform([data.location])[0]
    type_encoded = le_type.transform([data.property_type])[0]

    features = np.array([[
    
        loc_encoded,
        type_encoded,
        data.rooms,
        data.bathrooms,
        data.area_sqft,
        data.floor,
        data.year_built,
        data.has_pool,
        data.has_gym,
        data.has_parking,
        data.has_security,
        data.has_activity
    ]])

    predicted_price = model.predict(features)[0]

    return {
        "predicted_price_usd": round(float(predicted_price), 2),
        "status": "success"
    }

@app.get("/")
def root():
    return {"message": "Homey AI API is running!"}
@app.get("/check")
def check():
    return {
        "locations": list(le_location.classes_),
        "property_types": list(le_type.classes_)
    }