from pydantic import BaseModel , Field , field_validator , model_validator
from typing import Optional
from enum import Enum

from starlette.exceptions import HTTPException

import app

class PropertyType(str , Enum):
  apartment : "apartment"
  villa : "villa"
  bunglow : "bunglow"
  studio : "studio"

class FurnishingStatus(str , Enum):
  furnished : "furnished"
  semi_furnished : "semi_furnished"
  unfurnished : "unfurnished"

class HouseInput(BaseModel):

  size_sqft : float = Field(... , gt=0 , lt =100000 , description="Size of the house in square feet")

  num_bedrooms : int = Field(... , ge=1 , le=20 , description="Number of bedrooms in the house")

  num_bathrooms : int = Field(... , ge=1 , le=20 , description="Number of bathrooms in the house")

  floor_number : int = Field(... , ge=0 , le=150 , description="Floor number of the house")

  total_floors : int = Field(... , ge=1 , le=150 , description="Total number of floors in the building")

  age_of_property : int = Field(... , ge=0 , le=100 , description="Age of the property in years")

  # 2. optional fields
  distance_from_center_km : Optional[float] = Field(None , gt=0 , lt=200 , description="Distance of the property from the city center in kilometers")

  city : Optional[str] = Field(min_length=2 , max_length=50 , description="City where the property is located")

  # 3. using Enum here
  property_type : PropertyType= Field(description="Type of the property")

  furnishing_status : FurnishingStatus = Field(description="Furnishing status of the property")

  has_parking :bool= Field(default= False, description="Whether the property has parking facilities")

  has_gym : bool = Field(default= False, description="Whether the property has gym facilities")

  has_swimming_pool : bool = Field(default= False, description="Whether the property has swimming pool facilities")

  # 4. field validator 

  @field_validator('city')
  @classmethod
  def validate_city(cls, value:str) -> str:
    value = value.strip()

    if not value:
      raise ValueError("City name cannot be empty or just whitespace")
    return value.title()
  
  @field_validator('size_sqft')
  @classmethod
  def validate_size_sqft(cls, value:float) -> float:
    if value < 100:
      raise ValueError("Size of the house must be at least 100 sqft") 
    return value

  # 5. model validator

  @model_validator(mode="after")
  def validate_floor_logic(self) -> "HouseInput":
    if self.floor_number > self.total_floors:
      raise ValueError(
        f"floor_number ({self.floor_number}) cannot exceed "
        f"total_floors ({self.total_floors})"
      )
    return self
  
  @model_validator(mode="after")
  def validate_bathroom_bedroom(self)-> "HouseInput":
    if self.num_bathrooms > self.num_bedrooms * 2:
      raise ValueError(
        f"Number of bathrooms ({self.num_bathrooms}) seems too high."
        f"for ({self.num_bedrooms}) bedrooms "
      )
    return self
   
  # 6. model config
  model_config = {
    "json_schema_extra": {
      "example": {
        "size_sqft": 1500,
        "num_bedrooms": 3,
        "num_bathrooms": 2,
        "floor_number": 5,
        "total_floors": 10,
        "age_of_property": 5,
        "distance_from_center_km": 10.5,
        "city": "Mumbai",
        "property_type": "apartment",
        "furnishing_status": "furnished",
        "has_parking": True,
        "has_gym": True,
        "has_swimming_pool": False
      }
    }
  }

  # 7. output schema - what we send back to the client


class PricePrediction(BaseModel):
  predicted_price_inr : float = Field(description="Predicted price of the house in INR")

  price_per_sqft_inr : float = Field(description="Predicted price per square foot in INR")

  confidence_score : float = Field(ge=0.0 , le = 1.0,description="Confidence score of the prediction (0 to 1)")

  price_range_low : float = Field(description="Lower bound of the predicted price range in INR")

  price_range_high : float = Field(description="Upper bound of the predicted price range in INR")

  city :str = Field(description="City where the property is located")


  # 8. nested models

class Locationinfo(BaseModel):
  city : str 
  state : str
  avg_price_per_sqft : float
  market_trend : str = Field(description="rising/falling/stable")

class MarketResponse(BaseModel):
  location: Locationinfo
  total_listings : int
  last_updated : str


# 9. partial update schema 

class HouseUpdateInput(BaseModel):
  size_sqft: Optional[float]= None
  num_bedrooms: Optional[int]             = None
  num_bathrooms: Optional[int]            = None
  age_of_property: Optional[int]          = None
  furnishing_status: Optional[FurnishingStatus] = None
  has_parking: Optional[bool]             = None
  has_gym: Optional[bool]                 = None
  has_swimming_pool: Optional[bool]       = None

