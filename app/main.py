from fastapi import FastAPI , HTTPException

from app.schemas import(
    HouseInput,
    PricePrediction,
    MarketResponse,
    Locationinfo,
    HouseUpdateInput
)

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



@app.post("/predict" , response_model=PricePrediction)
def price_prediction(house: HouseInput):

    base_price = house.size_sqft * 5000

    # City multiplier
    city_multipliers = {
        "Mumbai": 3.0,
        "Delhi": 2.5,
        "Bangalore": 2.2,
        "Pune": 1.8,
        "Hyderabad": 1.7,
    }
    multiplier = city_multipliers.get(house.city, 1.5)

    # Adjustments
    bedroom_bonus    = house.num_bedrooms * 50000
    age_discount     = house.age_of_property * 20000
    parking_bonus    = 200000 if house.has_parking else 0
    gym_bonus        = 150000 if house.has_gym else 0
    pool_bonus       = 300000 if house.has_swimming_pool else 0

    furnishing_bonus = {
        "furnished": 300000,
        "semi-furnished": 150000,
        "unfurnished": 0
    }.get(house.furnishing_status.value, 0)

    # Final price
    predicted_price = (
        (base_price * multiplier)
        + bedroom_bonus
        - age_discount
        + parking_bonus
        + gym_bonus
        + pool_bonus
        + furnishing_bonus
    )

    predicted_price = max(predicted_price, 500000)  # floor price

    return PricePrediction(
        predicted_price_inr=round(predicted_price, 2),
        price_per_sqft=round(predicted_price / house.size_sqft, 2),
        confidence_score=0.82,
        price_range_low=round(predicted_price * 0.90, 2),
        price_range_high=round(predicted_price * 1.10, 2),
        city=house.city,
        property_type=house.property_type.value
    )


@app.get("/market/{city}", response_model=MarketResponse)
def get_market_info(city:str):
    market_data = {
        "Mumbai": {
            "state": "Maharashtra",
            "avg_price_per_sqft": 15000,
            "market_trend": "rising",
            "total_listings": 4523
        },
        "Pune": {
            "state": "Maharashtra",
            "avg_price_per_sqft": 8000,
            "market_trend": "stable",
            "total_listings": 2100
        },
        "Bangalore": {
            "state": "Karnataka",
            "avg_price_per_sqft": 9500,
            "market_trend": "rising",
            "total_listings": 3800
        }
    }
    if city not in market_data:
        raise HTTPException(status_code=404, 
                            detail=f"Market data for city '{city}' not found." f"Available ciites: {list(market_data.keys())}")
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