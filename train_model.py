import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# 1. Create synthetic dataset
np.random.seed(42)
NUM_SAMPLES = 5000

print("🏗️  Generating synthetic house data...")

# Raw features
size_sqft              = np.random.uniform(300, 5000, NUM_SAMPLES)
num_bedrooms           = np.random.randint(1, 6, NUM_SAMPLES)
num_bathrooms          = np.random.randint(1, 5, NUM_SAMPLES)
floor_number           = np.random.randint(0, 30, NUM_SAMPLES)
total_floors           = floor_number + np.random.randint(0, 20, NUM_SAMPLES)
age_of_property        = np.random.randint(0, 40, NUM_SAMPLES)
distance_from_center   = np.random.uniform(1, 50, NUM_SAMPLES)
has_parking            = np.random.randint(0, 2, NUM_SAMPLES)
has_gym                = np.random.randint(0, 2, NUM_SAMPLES)
has_swimming_pool      = np.random.randint(0, 2, NUM_SAMPLES)

cities = np.random.choice(
    ["Mumbai", "Delhi", "Bangalore", "Pune", "Hyderabad"],
    NUM_SAMPLES
)

property_types = np.random.choice(
    ["apartment", "villa", "bungalow", "studio"],
    NUM_SAMPLES
)

furnishing = np.random.choice(
    ["furnished", "semi-furnished", "unfurnished"],
    NUM_SAMPLES
)

# 2. create realistic prices
city_multiplier = {
    "Mumbai":    3.0,
    "Delhi":     2.5,
    "Bangalore": 2.2,
    "Pune":      1.8,
    "Hyderabad": 1.7,
}

property_multiplier = {
    "villa":     1.5,
    "bungalow":  1.4,
    "apartment": 1.0,
    "studio":    0.7,
}

furnishing_bonus = {
    "furnished":      300000,
    "semi-furnished": 150000,
    "unfurnished":    0,
}

prices = []
for i in range(NUM_SAMPLES):
    base     = size_sqft[i] * 5000
    cm       = city_multiplier[cities[i]]
    pm       = property_multiplier[property_types[i]]
    fb       = furnishing_bonus[furnishing[i]]
    bed_b    = num_bedrooms[i] * 50000
    age_d    = age_of_property[i] * 15000
    park_b   = has_parking[i] * 200000
    gym_b    = has_gym[i] * 150000
    pool_b   = has_swimming_pool[i] * 300000
    dist_d   = distance_from_center[i] * 10000

    # Add realistic random noise (±10%)
    noise    = np.random.uniform(0.90, 1.10)

    price = (
        base * cm * pm
        + fb + bed_b - age_d
        + park_b + gym_b + pool_b - dist_d
    ) * noise

    prices.append(max(price, 500000))   # floor at 5 lakh

# 3. Create DataFrame
df = pd.DataFrame({
    "size_sqft":            size_sqft,
    "num_bedrooms":         num_bedrooms,
    "num_bathrooms":        num_bathrooms,
    "floor_number":         floor_number,
    "total_floors":         total_floors,
    "age_of_property":      age_of_property,
    "distance_from_center": distance_from_center,
    "has_parking":          has_parking,
    "has_gym":              has_gym,
    "has_swimming_pool":    has_swimming_pool,
    "city":                 cities,
    "property_type":        property_types,
    "furnishing_status":    furnishing,
    "price":                prices,
})

# 4. Encode categorical features
le_city = LabelEncoder()
le_property = LabelEncoder()
le_furnishing = LabelEncoder()

df["city_encoded"] = le_city.fit_transform(df["city"])
df["property_encoded"] = le_property.fit_transform(df["property_type"])
df["furnished_encoded"] = le_furnishing.fit_transform(df["furnishing_status"])

# 5. Prepare features and target
feature_cols = [
    "size_sqft", "num_bedrooms", "num_bathrooms", "floor_number",
    "total_floors", "age_of_property", "distance_from_center",
    "has_parking", "has_gym", "has_swimming_pool",
    "city_encoded", "property_encoded", "furnished_encoded"
]

X = df[feature_cols]
y = df["price"]

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Train Random Forest model

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1,
    min_samples_split=5,
    min_samples_leaf=2
)

model.fit(X_train, y_train)

# 8. Evaluate model
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

r2= r2_score(y_test, y_pred)

print(f"    MAE : ₹{mae:,.0f}")
print(f"    R²  : {r2:.4f}")

# 9. Feature importance
importances = pd.Series(
    model.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

print(f"\n🔍  Feature Importances:")
for feat, imp in importances.items():
    bar = "█" * int(imp * 50)
    print(f"    {feat:<25} {bar} {imp:.4f}")

# 10. Save model and encoders
os.makedirs("models", exist_ok=True)

print("\n💾  Saving model and encoders...")

joblib.dump(model,              "models/house_price_model.joblib")
joblib.dump(le_city,            "models/le_city.joblib")
joblib.dump(le_property,        "models/le_property.joblib")
joblib.dump(le_furnishing,      "models/le_furnishing.joblib")
joblib.dump(feature_cols,       "models/feature_columns.joblib")

print("✅  Saved:")
print("    models/house_price_model.joblib")
print("    models/le_city.joblib")
print("    models/le_property.joblib")
print("    models/le_furnish.joblib")
print("    models/feature_columns.joblib")
print("\n🎉  Training pipeline complete!")