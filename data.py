import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("dataset.csv")
df.columns = df.columns.str.strip()

df['District'] = df['District'].str.upper()
df['Season'] = df['Season'].str.capitalize()

# Prepare scaler
features = df[['Rainfall_mm', 'Temperature_K', 'Area_hectare', 'N', 'P', 'K']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


def get_best_match(input_data):
    Year = input_data['Year']
    District = input_data['District']
    Season = input_data['Season']

    Rainfall = input_data['Rainfall']
    Temperature = input_data['Temperature']
    Area = input_data['Area']
    N = input_data['N']
    P = input_data['P']
    K = input_data['K']

    # EXACT MATCH
    exact = df[
        (df['Year'] == Year) &
        (df['District'] == District) &
        (df['Season'] == Season)
    ]

    if not exact.empty:
        best = exact.sort_values(by='Yield_tph', ascending=False).iloc[0]
        return {
            "crop": best['Crop'],
            "yield": round(best['Yield_tph'], 2),
            "production": round(best['Production_tonnes'], 2),
            "mode": "Exact Dataset Match"
        }

    # CLOSEST MATCH
    input_vector = [[Rainfall, Temperature, Area, N, P, K]]
    scaled_input = scaler.transform(input_vector)

    distances = euclidean_distances(scaled_input, scaled_features)
    index = distances.argmin()

    best = df.iloc[index]

    return {
        "crop": best['Crop'],
        "yield": round(best['Yield_tph'], 2),
        "production": round(best['Production_tonnes'], 2),
        "mode": "Closest Dataset Match"
    }


# ANALYSIS FUNCTION
def analyze_yield(result, input_data):
    yield_val = result["yield"]

    N = input_data["N"]
    P = input_data["P"]
    K = input_data["K"]
    Rainfall = input_data["Rainfall"]
    Temperature = input_data["Temperature"]

    reasons = []
    tips = []


    # LOW YIELD
    if yield_val < 1.3:
        level = "Low Yield"

        # 🌱 Nutrient Issues
        if N < 50:
            reasons.append("Nitrogen deficiency reduces leaf growth and chlorophyll formation, limiting photosynthesis.")
            tips.append("Apply Nitrogen fertilizers like Urea in split doses to improve plant growth.")

        if P < 30:
            reasons.append("Low Phosphorus affects root development and energy transfer in plants.")
            tips.append("Use DAP or SSP fertilizers to enhance root strength and early growth.")

        if K < 30:
            reasons.append("Potassium deficiency weakens disease resistance and reduces crop quality.")
            tips.append("Apply MOP (Muriate of Potash) to improve stress tolerance and yield quality.")

        # 🌧 Climate Issues
        if Rainfall < 500:
            reasons.append("Insufficient rainfall leads to water stress, reducing nutrient absorption and crop growth.")
            tips.append("Adopt irrigation methods like drip or sprinkler systems for efficient water use.")

        if Rainfall > 2000:
            reasons.append("Excess rainfall may cause waterlogging and nutrient leaching.")
            tips.append("Improve drainage system to avoid root damage and nutrient loss.")

        if Temperature > 35:
            reasons.append("High temperature causes heat stress, affecting flowering and grain filling.")
            tips.append("Use mulching or shade nets to reduce soil temperature and moisture loss.")

        if Temperature < 15:
            reasons.append("Low temperature slows plant metabolism and delays growth.")
            tips.append("Choose cold-resistant crop varieties or adjust sowing time.")

        # General Issues
        if not reasons:
            reasons.append("Poor soil health or improper farming practices affecting crop productivity.")
            tips.append("Incorporate organic manure and compost to improve soil fertility and structure.")

        # Additional Smart Tips
        tips.extend([
            "Conduct soil testing before fertilizer application.",
            "Use certified high-yield seed varieties.",
            "Follow proper crop rotation to maintain soil nutrients.",
            "Control weeds to reduce competition for nutrients.",
            "Monitor pests and apply bio-pesticides if needed."
        ])


    # MEDIUM YIELD
    elif 1.3 <= yield_val <= 2.5:
        level = "Medium Yield"

        reasons = [
            "Moderate nutrient availability and acceptable climatic conditions.",
            "Suboptimal farming practices slightly limiting maximum yield potential."
        ]

        tips = [
            "Optimize NPK fertilizer ratio based on soil test results.",
            "Adopt precision farming techniques.",
            "Improve irrigation scheduling (avoid over/under watering).",
            "Use hybrid or improved seed varieties.",
            "Apply micronutrients like Zinc and Boron.",
            "Ensure timely sowing and harvesting.",
            "Use integrated pest management (IPM) techniques.",
            "Improve soil organic matter using compost or green manure."
        ]

    # HIGH YIELD
    else:
        level = "High Yield"

        reasons = [
            "Favorable soil fertility with balanced nutrients.",
            "Optimal climatic conditions supporting crop growth.",
            "Effective farming practices leading to high productivity."
        ]

        tips = [
            "Maintain current nutrient management practices.",
            "Continue regular soil testing to sustain fertility.",
            "Adopt precision agriculture tools for better monitoring.",
            "Use advanced irrigation systems for efficiency.",
            "Protect crops from pests and diseases proactively.",
            "Store harvested crops properly to avoid post-harvest losses.",
            "Consider crop diversification for long-term sustainability."
        ]

    return {
        "level": level,
        "reasons": reasons,
        "tips": tips
    }