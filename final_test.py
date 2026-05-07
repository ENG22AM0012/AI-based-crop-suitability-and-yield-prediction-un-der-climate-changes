# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import joblib


# 2. LOAD DATA + MODELS
df = pd.read_csv("dataset.csv")

df.columns = df.columns.str.strip()

df['District'] = df['District'].astype(str).str.strip().str.upper()
df['Season'] = df['Season'].astype(str).str.strip().str.capitalize()

crop_model = joblib.load("best_crop_model.pkl")
yield_model = joblib.load("best_yield_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")


# 3. USER INPUT
print("\n===== ENTER INPUT VALUES =====")

Year = int(input("Year: "))
District = input("District: ").strip().upper()
Season = input("Season (Kharif/Rabi/Summer): ").strip().capitalize()

Rainfall = float(input("Rainfall (mm): "))
Temperature = float(input("Temperature (Kelvin): "))
Area = float(input("Area (hectare): "))

N = float(input("Nitrogen (N): "))
P = float(input("Phosphorus (P): "))
K = float(input("Potassium (K): "))

result = df[
    (df['Year'] == Year) &
    (df['District'] == District) &
    (df['Season'] == Season) &
    (df['Rainfall_mm'].round(3) == round(Rainfall, 3)) &
    (df['Temperature_K'].round(3) == round(Temperature, 3)) &
    (df['Area_hectare'] == Area) &
    (df['N'] == N) &
    (df['P'] == P) &
    (df['K'] == K)
]

if not result.empty:
    print("\n===== EXACT DATASET RESULT =====")
    
    row = result.iloc[0]
    
    print(f"Crop        : {row['Crop']}")
    print(f"Production  : {row['Production_tonnes']}")
    print(f"Yield       : {row['Yield_tph']}")


# 6.  ML PREDICTION
else:
    print("\n Exact match not found → Using ML Prediction")

    # Validation
    if District not in label_encoders['District'].classes_:
        print("Invalid District")
        exit()

    if Season not in label_encoders['Season'].classes_:
        print("Invalid Season")
        exit()

    # Encoding
    District_enc = label_encoders['District'].transform([District])[0]
    Season_enc = label_encoders['Season'].transform([Season])[0]

    # Feature Engineering
    Rainfall_per_area = Rainfall / (Area + 1)
    NPK_sum = N + P + K
    NPK_ratio = N / (P + K + 1)
    Temp_Rainfall = Temperature * Rainfall
    Year_trend = Year - df['Year'].min()

    input_data = pd.DataFrame([{
        'Year': Year,
        'District': District_enc,
        'Season': Season_enc,
        'Rainfall_mm': Rainfall,
        'Temperature_K': Temperature,
        'Area_hectare': Area,
        'N': N,
        'P': P,
        'K': K,
        'Rainfall_per_area': Rainfall_per_area,
        'NPK_sum': NPK_sum,
        'NPK_ratio': NPK_ratio,
        'Temp_Rainfall': Temp_Rainfall,
        'Year_trend': Year_trend
    }])

    # Prediction
    crop_pred = crop_model.predict(input_data)[0]
    crop_name = label_encoders['Crop'].inverse_transform([crop_pred])[0]

    yield_pred_log = yield_model.predict(input_data)[0]
    yield_pred = np.expm1(yield_pred_log)

    production_pred = yield_pred * Area

    print("\n===== ML PREDICTION RESULT =====")
    print(f"Crop        : {crop_name}")
    print(f"Production  : {production_pred}")
    print(f"Yield       : {yield_pred}")
