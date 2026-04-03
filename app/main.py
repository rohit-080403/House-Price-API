from tracemalloc import start

from fastapi import FastAPI , HTTPException
from contextlib import asynccontextmanager

from fastapi.params import Depends

from app.schemas import(
    HouseInput,
    PricePrediction,
    MarketResponse,
    Locationinfo,
    HouseUpdateInput
)

from app.model_loader import house_model

from app.dependencies import (
    get_house_model,
    get_pagination,
    get_api_key,
    get_model_with_auth,
    validate_city
)

app = FastAPI(
  title="House Price Prediction",
  description ="An API to predict house prices using Ml model",
  version="1.0.0"
)

@app.on_event("startup")
def startup_event():
    house_model.load()

@app.get("/")
def root():
    return {"message": "Welcome to the House Price Prediction API!"}

@app.get("/health")
def health_check():
    return {
        "status":"healthy",
        "model_loaded": house_model.is_loaded,
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



@app.post("/predict" , response_model=PricePrediction)
def predict_prices(
    house_input: HouseInput,
    model = Depends(get_house_model) #<- injected dependency
):
    result = model.predict(house_input)
    return PricePrediction(**result)

# PREDICT SECURE - with auth + model together
@app.post("/predict/secure", response_model=PricePrediction)
def predict_secure(
    input: HouseInput,
    auth = Depends(get_model_with_auth) # <- chained dependency
):
    user_id = auth["user_id"]
    model = auth["model"]

    print(f"User {user_id} is making a prediction")

    result = model.predict(input)
    return PricePrediction(**result)
       

# LISTINGS ENDPOINT - with pagination DEPENDENCY
@app.get("/listings")
def get_listings(
    pagination = Depends(get_pagination) # <- injected pagination dependency
):
  """
    Endpoint to list house listings with pagination.
    Uses the get_pagination dependency to get page, page_size, and offset.
  """
  all_listings = [
    {"id": i, "city": "Mumbai", "price": 10000000*i}
    for i in range(1, 51) 
    ]
  
  #apply pagination using the offset 
  start  = pagination["offset"]
  end    = start + pagination["page_size"]
  paged  = all_listings[start:end]

  return {
        "page":         pagination["page"],
        "page_size":    pagination["page_size"],
        "total":        len(all_listings),
        "results":      paged
    }


@app.get("/market/{city}", response_model=MarketResponse)
def get_market_info(
    city= Depends(validate_city) # <- injected - validated !!!
): 
    market_data = {
        "Mumbai":    {"state": "Maharashtra", "avg_price_per_sqft": 15000, "market_trend": "rising",  "total_listings": 4523},
        "Pune":      {"state": "Maharashtra", "avg_price_per_sqft": 8000,  "market_trend": "stable",  "total_listings": 2100},
        "Bangalore": {"state": "Karnataka",   "avg_price_per_sqft": 9500,  "market_trend": "rising",  "total_listings": 3800},
        "Delhi":     {"state": "Delhi",       "avg_price_per_sqft": 12000, "market_trend": "rising",  "total_listings": 5100},
        "Hyderabad": {"state": "Telangana",   "avg_price_per_sqft": 7500,  "market_trend": "stable",  "total_listings": 2800},
    }

    data = market_data[city]

    return MarketResponse(
        location=Locationinfo(
            city=city,
            state=data["state"],
            avg_price_per_sqft=data["avg_price_per_sqft"],
            market_trend=data["market_trend"]
        ),
        total_listings=data["total_listings"],
    )

@app.patch("/listing/{listing_id}")
def update_listing(listing_id: int, update_data: HouseUpdateInput):

  updated_fields = update_data.model_dump(exclude_none =True)    

  if not updated_fields:
    raise HTTPException(status_code=400 , detail="No valid fields provided for update")

  return {
    "listing_id": listing_id,
    "updated_fields": updated_fields,
    "message": "Listing updated successfully"
  }       