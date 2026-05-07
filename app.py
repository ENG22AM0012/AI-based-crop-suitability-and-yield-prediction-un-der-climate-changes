from flask import Flask, render_template, request, redirect, session, send_file
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import joblib

# 🔥 PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

from data import get_best_match, analyze_yield

app = Flask(__name__)
app.secret_key = "secret123"


# ==============================
# DATABASE
# ==============================
def init_db():
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()


# ==============================
# LOAD MODELS
# ==============================
try:
    crop_model = joblib.load("best_crop_model.pkl")
    yield_model = joblib.load("best_yield_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
except Exception as e:
    print("⚠️ Model loading error:", e)


# ==============================
# ROUTES
# ==============================
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("users.db")
        cur = conn.cursor()
        cur.execute("INSERT INTO users (username,password) VALUES (?,?)",
                    (username, password))
        conn.commit()
        conn.close()

        return redirect("/login")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("users.db")
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username=? AND password=?",
                    (username, password))
        user = cur.fetchone()
        conn.close()

        if user:
            session["user"] = username
            return redirect("/predict")
        else:
            return "Invalid login"

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/")


# ==============================
# VIEW USERS
# ==============================
@app.route("/view_users")
def view_users():
    if "user" not in session:
        return redirect("/login")

    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("SELECT id, username FROM users")
    users_list = cur.fetchall()
    conn.close()

    return render_template("users.html", users=users_list)


# ==============================
# PREDICT
# ==============================
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect("/login")

    result = None
    analysis = None
    shap_path = None
    shap_text = []

    if request.method == "POST":
        try:
            Year = int(request.form["Year"])
            District = request.form["District"].strip().upper()
            Season = request.form["Season"].strip().capitalize()

            Rainfall = float(request.form["Rainfall"])
            Temperature = float(request.form["Temperature"])
            Area = float(request.form["Area"])

            N = float(request.form["N"])
            P = float(request.form["P"])
            K = float(request.form["K"])

            # ENCODE
            District_enc = label_encoders['District'].transform([District])[0]
            Season_enc = label_encoders['Season'].transform([Season])[0]

            # FEATURES
            NPK_sum = N + P + K
            NPK_ratio = N / (P + K + 1e-5)
            Rainfall_per_area = Rainfall / (Area + 1e-5)
            Temp_Rainfall = Temperature * Rainfall
            Year_trend = Year - 2000

            input_model = pd.DataFrame([{
                'Year': Year,
                'District': District_enc,
                'Season': Season_enc,
                'Rainfall_mm': Rainfall,
                'Temperature_K': Temperature,
                'Area_hectare': Area,
                'N': N,
                'P': P,
                'K': K,
                'NPK_sum': NPK_sum,
                'NPK_ratio': NPK_ratio,
                'Rainfall_per_area': Rainfall_per_area,
                'Temp_Rainfall': Temp_Rainfall,
                'Year_trend': Year_trend
            }])

            # GRAPH
            if not os.path.exists("static"):
                os.makedirs("static")

            values = input_model.iloc[0].values
            importance = values / (np.max(values) + 1e-5)

            plt.figure(figsize=(8, 5))
            plt.barh(input_model.columns, importance, color='skyblue')
            plt.xlabel("Impact")
            plt.title("Feature Contribution")

            shap_path = "static/shap.png"
            plt.savefig(shap_path, bbox_inches='tight')
            plt.close()

            # TEXT
            shap_text = []
            for name, val in zip(input_model.columns, importance):
                if val > 0:
                    shap_text.append(f"{name} increased prediction")
                else:
                    shap_text.append(f"{name} decreased prediction")

            # DATASET RESULT
            input_data = {
                "Year": Year,
                "District": District,
                "Season": Season,
                "Rainfall": Rainfall,
                "Temperature": Temperature,
                "Area": Area,
                "N": N,
                "P": P,
                "K": K
            }

            result = get_best_match(input_data)
            analysis = analyze_yield(result, input_data)

            # 🔥 STORE FOR PDF
            session["result"] = result
            session["analysis"] = analysis
            session["shap_text"] = shap_text
            session["shap_path"] = shap_path
            session["input_data"] = input_data

        except Exception as e:
            print("❌ ERROR:", e)
            return f"Error: {e}"

    return render_template(
        "predict.html",
        result=result,
        analysis=analysis,
        shap_path=shap_path,
        shap_text=shap_text
    )


# ==============================
# DOWNLOAD PDF
# ==============================
@app.route("/download")
def download():
    if "user" not in session:
        return redirect("/login")

    result = session.get("result")
    analysis = session.get("analysis")
    shap_text = session.get("shap_text")
    shap_path = session.get("shap_path")
    input_data = session.get("input_data")

    if not result:
        return "No data to download"

    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Crop Prediction Report", styles['Title']))
    content.append(Spacer(1, 12))

    content.append(Paragraph("Input Values:", styles['Heading2']))
    for k, v in input_data.items():
        content.append(Paragraph(f"{k}: {v}", styles['Normal']))

    content.append(Spacer(1, 12))

    content.append(Paragraph("Result:", styles['Heading2']))
    content.append(Paragraph(f"Crop: {result['crop']}", styles['Normal']))
    content.append(Paragraph(f"Yield: {result['yield']}", styles['Normal']))
    content.append(Paragraph(f"Production: {result['production']}", styles['Normal']))

    content.append(Spacer(1, 12))

    content.append(Paragraph("Analysis:", styles['Heading2']))
    for r in analysis["reasons"]:
        content.append(Paragraph(f"- {r}", styles['Normal']))
    for t in analysis["tips"]:
        content.append(Paragraph(f"- {t}", styles['Normal']))

    content.append(Spacer(1, 12))

    content.append(Paragraph("Explanation:", styles['Heading2']))
    for s in shap_text:
        content.append(Paragraph(f"- {s}", styles['Normal']))

    content.append(Spacer(1, 12))

    if shap_path and os.path.exists(shap_path):
        content.append(Paragraph("SHAP Graph:", styles['Heading2']))
        content.append(Image(shap_path, width=400, height=300))

    doc.build(content)

    return send_file("report.pdf", as_attachment=True)


# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    app.run(debug=True)