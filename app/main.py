from fastapi import FastAPI

app = FastAPI(
  title="House Price Prediction",
  description ="An API to predict house prices using Ml model",
  version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "Welcome to the House Price Prediction API!"}

@app.get("/health")
def health_check():
    return {
        "status":"healthy",
        "api_version" : "1.0.0"
    }

@app.get("/city/{city_name}")
def get_city_info(city_name:str):
    city_data={
        "mumbai": {"avg_price_per_sqft": 15000, "tier": 1},
        "pune": {"avg_price_per_sqft": 12000, "tier": 2},
        "nagpur": {"avg_price_per_sqft": 9000, "tier": 3},
    }
    city = city_name.lower()

    if city not in city_data:
        return {"error": f"City '{city_name}' not found in the database."}
    return {"city": city_name,
            "data": city_data[city]
    }

@app.get("/estimate")
def rough_estimate(size:float , rooms : int =1):
    price = size * 5000 + rooms * 50000
    return {"size_sqft": size,
            "rooms": rooms,
            "rough_estimate_int": int(price)
    }