# app/model_loader.py

import joblib
import numpy as np
from pathlib import Path

# -----------------------------------------------
# WHY A SEPARATE LOADER FILE?
# -----------------------------------------------
# We don't want to load the model inside main.py
# because:
# 1. Separation of concerns — loading logic is separate
# 2. Easy to swap models later
# 3. Can be tested independently
# 4. Will be used by lifespan events in Step 5

# Base path — always relative to project root
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


class HousePriceModel:
    """
    Wrapper class around the trained sklearn model.
    Handles loading, preprocessing, and prediction.
    """

    def __init__(self):
        self.model           = None
        self.le_city         = None
        self.le_property     = None
        self.le_furnish      = None
        self.feature_columns = None
        self.is_loaded       = False


    def load(self):
        """Load model and encoders from disk."""
        print("📦  Loading model from disk...")

        self.model           = joblib.load(MODEL_DIR / "house_price_model.joblib")
        self.le_city         = joblib.load(MODEL_DIR / "le_city.joblib")
        self.le_property     = joblib.load(MODEL_DIR / "le_property.joblib")
        self.le_furnish      = joblib.load(MODEL_DIR / "le_furnishing.joblib")
        self.feature_columns = joblib.load(MODEL_DIR / "feature_columns.joblib")

        self.is_loaded = True
        print("✅  Model loaded successfully!")


    def predict(self, house_input) -> dict:
        """
        Takes a validated HouseInput object,
        preprocesses it, runs prediction,
        and returns a result dict.
        """

        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # -----------------------------------------------
        # ENCODE CATEGORICAL FEATURES
        # -----------------------------------------------
        # We must use the SAME encoders used during training
        # Otherwise "Mumbai" might map to a different number!

        try:
            city_encoded = self.le_city.transform([house_input.city])[0]
        except ValueError:
            # City not seen during training — use most common encoding
            city_encoded = 0

        try:
            property_encoded = self.le_property.transform(
                [house_input.property_type.value]
            )[0]
        except ValueError:
            property_encoded = 0

        try:
            furnish_encoded = self.le_furnish.transform(
                [house_input.furnishing_status.value]
            )[0]
        except ValueError:
            furnish_encoded = 0

        # -----------------------------------------------
        # BUILD FEATURE ARRAY
        # -----------------------------------------------
        # Order MUST match FEATURE_COLUMNS from training
        # Even one wrong order = completely wrong predictions

        distance = house_input.distance_from_center_km \
                   if house_input.distance_from_center_km else 10.0

        features = np.array([[
            house_input.size_sqft,
            house_input.num_bedrooms,
            house_input.num_bathrooms,
            house_input.floor_number,
            house_input.total_floors,
            house_input.age_of_property,
            distance,
            int(house_input.has_parking),
            int(house_input.has_gym),
            int(house_input.has_swimming_pool),
            city_encoded,
            property_encoded,
            furnish_encoded,
        ]])

        # -----------------------------------------------
        # RUN PREDICTION
        # -----------------------------------------------
        predicted_price = float(self.model.predict(features)[0])
        predicted_price = max(predicted_price, 500000)

        # -----------------------------------------------
        # CONFIDENCE SCORE
        # -----------------------------------------------
        # RandomForest = many trees
        # Each tree gives its own prediction
        # If all trees agree → high confidence
        # If trees disagree → low confidence

        tree_predictions = np.array([
            tree.predict(features)[0]
            for tree in self.model.estimators_
        ])

        std_dev  = np.std(tree_predictions)
        cv       = std_dev / predicted_price        # coefficient of variation
        confidence = float(max(0.0, min(1.0, 1 - cv)))

        return {
            "predicted_price_inr": round(predicted_price, 2),
            "price_per_sqft_inr":  round(predicted_price / house_input.size_sqft, 2),
            "confidence_score":    round(confidence, 4),
            "price_range_low":     round(predicted_price * 0.90, 2),
            "price_range_high":    round(predicted_price * 1.10, 2),
            "city":                house_input.city,
            "property_type":       house_input.property_type.value,
        }


# -----------------------------------------------
# GLOBAL INSTANCE
# -----------------------------------------------
# One instance shared across the entire app
# Loaded once at startup (Step 5 — lifespan events)

house_model = HousePriceModel()