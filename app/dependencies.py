from fastapi import Depends , HTTPException , Header , Query , Path
from app.model_loader import house_model
from typing import Optional

# 1. basic dependency
def get_house_model():
  """Dependency that provides the loaded ML model.
    Raises 503 if model isn't loaded yet 
    """
  if not house_model.is_loaded:
    raise HTTPException(status_code=503, detail="Model is not loaded yet. Please try again later.")
  return house_model

# 2. dependency with parameters
def get_pagination(
    page : int = Query(1, ge=1, description="Page number (starting from 1)"),
    page_size : int = Query(10, ge=1, le=100, description="Number of items per page (1-100)")
):
  """
    Dependency that provides pagination parameters.
    Reusable across any listing endpoint.
    """
  return {
    "page": page,
    "page_size": page_size,
    "offset": (page - 1) * page_size
  }
    
# 3 . header dependency - read API key from headers
valid_api_keys = {"mysecretapikey123" : "user1",
                  "devkey456" : "user2",
                  "testkey789" : "user3"
}

def get_api_key(x_api_key: Optional[str] = Header(None)):
  """
    Dependency that validates API key from request header.
    Returns the user_id associated with the key.
    """
  if x_api_key is None:
    raise HTTPException(status_code=401, detail="API key missing")
  
  if x_api_key not in valid_api_keys:
    raise HTTPException(status_code=403, detail="Invalid API key")
  
  return valid_api_keys[x_api_key]

# 4. chained dependencies - depends on another dependency
def get_model_with_auth(
    user_id : str = Depends(get_api_key),
    model : object = Depends(get_house_model)
):
  """
    Chained dependency — combines auth + model.
    Returns both so the route has everything it needs.
    """
  return {
    "user_id": user_id,
    "model": model
  }

# 5. city validator dependency
supported_cities = {"mumbai", "pune", "nagpur", "delhi", "bangalore", "hyderabad"}

def validate_city(city_name: str = Path(..., alias="city")):
  """
    Dependency that validates and normalizes city names.
    Reusable across any route that takes a city.
    """
  normalize = city_name.strip().lower()

  if normalize not in supported_cities:
    raise HTTPException(
      status_code=404,
      detail={        
        "error":"City not supported",
        "requested_city": city_name,
        "supported_cities": sorted(supported_cities)
        }
     )
  return normalize.title()   