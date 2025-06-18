# Global Echo - Market Predictor (Streamlit App)

This Streamlit app predicts the **NASDAQ opening gap direction** (Up / Down / Flat) using an AI model trained on global market data and overnight price movements.

It is containerized with Docker for easy, platform-independent deployment.

It is also published on Streamlit Cloud [here](https://overnight-gap-predictor.streamlit.app/)

---

## How to Run Locally

### 1. Build the Docker image

```bash
cd Docker
docker build -t market-predictor-app .
```

### 2. Run the app

```bash
docker run -p 8501:8501 market-predictor-app
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## How it Works

- **Model**: XGBoost classifier  
- **Features**: Percent changes in global indices (e.g. DAX, FTSE, Nikkei, etc.)  
- **Target**: Whether NASDAQ opens **Up**, **Down**, or **Flat**  
- **Visuals**:
  - Prediction confidence bar chart
  - Historical NASDAQ candlestick chart  
- **Data**: Trained on historical data up to **February 2025**

---

## 📁 Included Files

- `app.py` – Streamlit interface  
- `xgb_best_model.pkl` – Trained XGBoost model  
- `best_model.pkl` – Label encoder  
- `Features.csv` / `Target.csv` – Cleaned features and targets  
- `Data/NASDAQ100.csv` – For visualizations  
- `Dockerfile` – Container configuration  
- `requirements.txt` – Python dependencies  
- `.dockerignore` – Prevents build clutter  
- `README.md` – This file

---

## ⚠️ Disclaimer

This app uses a machine learning model and is intended for educational and demonstration purposes only.  
**Do not use it to make real financial or trading decisions.**
