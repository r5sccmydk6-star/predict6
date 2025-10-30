from flask import Flask, render_template, request, redirect, url_for, session
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64, logging, traceback, os

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

# -------- Simple In-Memory User Storage --------
users = {"admin": "admin123"}  # default user

# -------- Logging setup --------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("error.log"), logging.StreamHandler()]
)

# -------- RSI Calculation --------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# -------- Download Data Safely --------
def safe_download(ticker, period="5y", interval="1d"):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
        if data is None or data.empty:
            data = yf.download(ticker + ".NS", period=period, interval=interval, progress=False, threads=False)
        return data.dropna() if data is not None else pd.DataFrame()
    except Exception as e:
        logging.error(f"Download error for {ticker}: {e}")
        return pd.DataFrame()


# ======================== AUTH ROUTES ========================
@app.route("/")
def index():
    if "user" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username in users and users[username] == password:
            session["user"] = username
            return redirect(url_for("dashboard"))
        else:
            error = "Invalid username or password."
    return render_template("login.html", error=error)


@app.route("/register", methods=["GET", "POST"])
def register():
    error = None
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username in users:
            error = "Username already exists."
        else:
            users[username] = password
            return redirect(url_for("login"))
    return render_template("register.html", error=error)


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


# ======================== DASHBOARD ========================
@app.route("/dashboard", methods=["GET"])
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))

    ticker = request.args.get("ticker", "").strip().upper()
    algo = request.args.get("algo", "Linear Regression")

    if not ticker:
        return render_template("dashboard.html", username=session["user"])

    try:
        # Step 1: Download data
        data = safe_download(ticker)
        if data.empty:
            return render_template("dashboard.html", username=session["user"], error="No data found.")

        if len(data) < 60:
            return render_template("dashboard.html", username=session["user"], error="Not enough data to train model.")

        # Step 2: Indicators
        data["RSI"] = compute_rsi(data["Close"])
        data["MA20"] = data["Close"].rolling(window=20).mean()
        data["EMA20"] = data["Close"].ewm(span=20, adjust=False).mean()
        data.dropna(inplace=True)

        # Step 3: Features
        X = data[["Open", "High", "Low", "Volume", "RSI", "MA20", "EMA20"]]
        y = data["Close"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Step 4: Choose Algorithm
        if algo == "Decision Tree":
            model = DecisionTreeRegressor(random_state=42)
        elif algo == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = LinearRegression()

        # Step 5: Train
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        y_test = np.array(y_test).flatten()
        pred = np.array(pred).flatten()

        # Step 6: Plot
        plt.figure(figsize=(10, 5))
        plt.plot(y_test[:100], label="Actual", color="skyblue")
        plt.plot(pred[:100], label="Predicted", color="orange")
        plt.legend()
        plt.title(f"{ticker} Prediction ({algo})")
        plt.tight_layout()

        img = io.BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        # Step 7: Next Day Prediction
        next_day_pred = float(model.predict(X.iloc[-1].values.reshape(1, -1))[0])
        current_price = float(y.iloc[-1])
        diff = next_day_pred - current_price
        direction = "rise" if diff > 0 else "fall"

        conclusion = f"Predicted next-day close: ${next_day_pred:.2f}. Current: ${current_price:.2f}. Expected {direction} of ${abs(diff):.2f}."

        # Advice + Warning
        crash_threshold = current_price * 0.95
        warning = None
        if next_day_pred < crash_threshold:
            warning = "⚠️ Model predicts a 5%+ drop! Consider holding or selling."

        if diff > 0:
            advice_class = "safe"
            advice_text = "🟢 Safe to Buy — Uptrend expected!"
        else:
            advice_class = "risky"
            advice_text = "🔴 Risky to Buy — Possible downtrend ahead."

        # Summary
        avg_actual = float(np.mean(y_test[-10:]))
        avg_pred = float(np.mean(pred[-10:]))
        accuracy = 100 - (abs(avg_actual - avg_pred) / avg_actual * 100)
        summary = f"📊 Model: {algo} | Accuracy ≈ {accuracy:.2f}%"

        table_data = list(zip(y_test[-10:], pred[-10:]))

        return render_template(
            "dashboard.html",
            username=session["user"],
            ticker=ticker,
            algo=algo,
            plot_url=plot_url,
            conclusion=conclusion,
            warning=warning,
            advice_text=advice_text,
            advice_class=advice_class,
            summary=summary,
            table_data=table_data
        )

    except Exception as e:
        logging.error(traceback.format_exc())
        return render_template("dashboard.html", username=session["user"], error=str(e))


# ======================== RUN APP ========================
if __name__ == "__main__":
    app.run(debug=True)
