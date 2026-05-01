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
all_properties = [

    # ──────────────────────────────────────────────
    # 💰 Budget Range: $50,000 – $80,000
    # ──────────────────────────────────────────────
    {"image": "https://images.unsplash.com/photo-1555041469-a586c61ea9bc", "price": 52000},  # living room sofa
    {"image": "https://images.unsplash.com/photo-1484101403633-562f891dc89a", "price": 54000},  # simple bedroom
    {"image": "https://images.unsplash.com/photo-1556909114-f6e7ad7d3136", "price": 55000},  # small kitchen
    {"image": "https://images.unsplash.com/photo-1583847268964-b28dc8f51f92", "price": 56000},  # cozy living room
    {"image": "https://images.unsplash.com/photo-1513694203232-719a280e022f", "price": 57000},  # simple dining
    {"image": "https://images.unsplash.com/photo-1560448204-603b3fc33ddc", "price": 58000},  # studio room
    {"image": "https://images.unsplash.com/photo-1524758631624-e2822e304c36", "price": 59000},  # bedroom white
    {"image": "https://images.unsplash.com/photo-1536349788264-1b816db3cc13", "price": 60000},  # small bathroom
    {"image": "https://images.unsplash.com/photo-1493809842364-78817add7ffb", "price": 61000},  # cozy corner
    {"image": "https://images.unsplash.com/photo-1522708323590-d24dbb6b0267", "price": 62000},  # apartment interior
    {"image": "https://images.unsplash.com/photo-1502672260266-1c1ef2d93688", "price": 63000},  # loft bedroom
    {"image": "https://images.unsplash.com/photo-1616594039964-ae9021a400a0", "price": 65000},  # white bedroom
    {"image": "https://images.unsplash.com/photo-1598928506311-c55ded91a20c", "price": 66000},  # bedroom wooden
    {"image": "https://images.unsplash.com/photo-1560185007-cde436f6a4d0", "price": 67000},  # kitchen simple
    {"image": "https://images.unsplash.com/photo-1505691938895-1758d7feb511", "price": 68000},  # dining table
    {"image": "https://images.unsplash.com/photo-1484154218962-a197022b5858", "price": 69000},  # kitchen countertop
    {"image": "https://images.unsplash.com/photo-1600210492486-724fe5c67fb0", "price": 70000},  # living area
    {"image": "https://images.unsplash.com/photo-1574643156929-51fa098b0394", "price": 71000},  # bedroom minimal
    {"image": "https://images.unsplash.com/photo-1568605114967-8130f3a36994", "price": 73000},  # house interior
    {"image": "https://images.unsplash.com/photo-1571508601891-ca5e7a713859", "price": 75000},  # bathroom tiles
    {"image": "https://images.unsplash.com/photo-1556020685-ae41abfc9365", "price": 77000},  # living room rug
    {"image": "https://images.unsplash.com/photo-1600566753190-17f0baa2a6c3", "price": 79000},  # bedroom calm

    # ──────────────────────────────────────────────
    # 💰💰 Mid Range: $80,000 – $130,000
    # ──────────────────────────────────────────────
    {"image": "https://images.unsplash.com/photo-1600121848594-d8644e57abab", "price": 82000},  # modern living room
    {"image": "https://images.unsplash.com/photo-1556228453-efd6c1ff04f6", "price": 84000},  # modern kitchen
    {"image": "https://images.unsplash.com/photo-1600607686527-6fb886090705", "price": 86000},  # elegant bathroom
    {"image": "https://images.unsplash.com/photo-1631049307264-da0ec9d70304", "price": 88000},  # hotel-like bedroom
    {"image": "https://images.unsplash.com/photo-1618221195710-dd6b41faaea6", "price": 90000},  # stylish bedroom
    {"image": "https://images.unsplash.com/photo-1615874959474-d609969a20ed", "price": 92000},  # modern kitchen island
    {"image": "https://images.unsplash.com/photo-1600047508788-786f3865b59b", "price": 94000},  # open living space
    {"image": "https://images.unsplash.com/photo-1588854337221-4cf9fa96059c", "price": 95000},  # bright dining
    {"image": "https://images.unsplash.com/photo-1616137466211-f939a420be84", "price": 97000},  # modern bathroom
    {"image": "https://images.unsplash.com/photo-1600573472592-401b489a3cdc", "price": 99000},  # cozy reading nook
    {"image": "https://images.unsplash.com/photo-1586023492125-27b2c045efd7", "price": 100000}, # living room grey
    {"image": "https://images.unsplash.com/photo-1564078516393-cf04bd966897", "price": 105000}, # stylish dining room
    {"image": "https://images.unsplash.com/photo-1600566752355-35792bedcfea", "price": 107000}, # master bedroom
    {"image": "https://images.unsplash.com/photo-1600585154526-990dced4db0d", "price": 109000}, # open kitchen modern
    {"image": "https://images.unsplash.com/photo-1507089947368-19c1da9775ae", "price": 112000}, # home office
    {"image": "https://images.unsplash.com/photo-1600210491892-03d54c0aaf87", "price": 115000}, # luxury bathroom sink
    {"image": "https://images.unsplash.com/photo-1560185008-b033106af5c3", "price": 117000}, # white kitchen cabinets
    {"image": "https://images.unsplash.com/photo-1567225557594-88d73e55f2cb", "price": 119000}, # bedroom curtains
    {"image": "https://images.unsplash.com/photo-1618219908412-a29a1bb7b86e", "price": 124000}, # aesthetic bedroom
    {"image": "https://images.unsplash.com/photo-1600607687920-4e2a09cf159d", "price": 127000}, # bathroom marble

    # ──────────────────────────────────────────────
    # 💰💰💰 Premium Range: $130,000 – $200,000
    # ──────────────────────────────────────────────
    {"image": "https://images.unsplash.com/photo-1616486338812-3dadae4b4ace", "price": 132000}, # luxury living
    {"image": "https://images.unsplash.com/photo-1600047508006-da0f81d59b40", "price": 136000}, # villa living room
    {"image": "https://images.unsplash.com/photo-1600566753086-00f18fb6b3ea", "price": 139000}, # premium bedroom
    {"image": "https://images.unsplash.com/photo-1600607687644-aac4c3eac7f4", "price": 142000}, # luxury bathroom
    {"image": "https://images.unsplash.com/photo-1615529328331-f8917597711f", "price": 145000}, # luxury kitchen
    {"image": "https://images.unsplash.com/photo-1631049552057-403cdb8f0658", "price": 148000}, # suite bedroom
    {"image": "https://images.unsplash.com/photo-1600047509807-ba8f99d2cdde", "price": 151000}, # designer living
    {"image": "https://images.unsplash.com/photo-1600210492493-0946911123ea", "price": 155000}, # spa bathroom
    {"image": "https://images.unsplash.com/photo-1573052905904-34ad8c27f0cc", "price": 158000}, # elegant kitchen
    {"image": "https://images.unsplash.com/photo-1600566752734-2a0cd69e3e0f", "price": 161000}, # penthouse bedroom
    {"image": "https://images.unsplash.com/photo-1571055107559-3e67626fa8be", "price": 165000}, # luxury dining
    {"image": "https://images.unsplash.com/photo-1600585154340-be6161a56a0c", "price": 168000}, # villa interior
    {"image": "https://images.unsplash.com/photo-1560448075-bb485b1e3a1a", "price": 175000}, # designer bathroom
    {"image": "https://images.unsplash.com/photo-1582063289852-62e3ba2747f8", "price": 178000}, # modern fireplace
    {"image": "https://images.unsplash.com/photo-1597218868981-1b68e15f0065", "price": 181000}, # high-end kitchen
    {"image": "https://images.unsplash.com/photo-1560449752-3fd4bdbe3093", "price": 185000}, # master suite
    {"image": "https://images.unsplash.com/photo-1600047509782-20d39509f26d", "price": 188000}, # luxury lounge
    {"image": "https://images.unsplash.com/photo-1618221469555-7f3ad97540d6", "price": 192000}, # premium interior
    {"image": "https://images.unsplash.com/photo-1600566753376-12c8ab7fb75b", "price": 196000}, # villa bedroom luxury
    {"image": "https://images.unsplash.com/photo-1583845112203-29329902332e", "price": 199000}, # ultra bedroom

    # ──────────────────────────────────────────────
    # 👑 Luxury Range: $200,000+
    # ──────────────────────────────────────────────
    {"image": "https://images.unsplash.com/photo-1613490493576-7fde63acd811", "price": 210000}, # luxury villa
    {"image": "https://images.unsplash.com/photo-1512917774080-9991f1c4c750", "price": 225000}, # mansion interior
    {"image": "https://images.unsplash.com/photo-1580587771525-78b9dba3b914", "price": 245000}, # luxury home
    {"image": "https://images.unsplash.com/photo-1558618666-fcd25c85cd64", "price": 255000}, # ultra luxury kitchen
    {"image": "https://images.unsplash.com/photo-1564013799919-ab600027ffc6", "price": 265000}, # grand bedroom
    {"image": "https://images.unsplash.com/photo-1600121848594-d8644e57abab", "price": 275000}, # palace living
    {"image": "https://images.unsplash.com/photo-1618219740975-d40978bb7378", "price": 285000}, # grand suite
    {"image": "https://images.unsplash.com/photo-1560448204-603b3fc33ddc", "price": 295000}, # ultra premium
    {"image": "https://images.unsplash.com/photo-1600566753190-17f0baa2a6c3", "price": 310000}, # infinity luxury
]


@app.get("/properties/search")
def get_similar(price: float, location: str):
    margin = 20000  # نطاق ±20,000

    filtered = [
        p for p in all_properties
        if (price - margin) <= p["price"] <= (price + margin)
    ]

    # إذا النتائج قليلة، وسّع النطاق تلقائياً
    if len(filtered) < 4:
        margin = 40000
        filtered = [
            p for p in all_properties
            if (price - margin) <= p["price"] <= (price + margin)
        ]

    return {
        "featured": filtered[:5], 
        "popular": filtered[5:]   
    }