import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from xgboost import XGBClassifier
import plotly.graph_objects as go
from datetime import date

# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: Path, enc_path: Path):
    model = joblib.load(model_path)
    enc = joblib.load(enc_path)
    classes = list(enc.classes_) if hasattr(enc, "classes_") else list(enc)
    return model, classes

@st.cache_data(show_spinner=False)
def load_csv(path: Path):
    df = pd.read_csv(path)
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    dt_vals  = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    dt_vals  = dt_vals.dt.tz_convert(None).dt.normalize()
    df[date_col] = dt_vals
    df.set_index(date_col, inplace=True)
    return df

@st.cache_data(show_spinner=False)
def predict_row(_model, class_names: list[str], row: pd.Series):
    probs = _model.predict_proba(row.values.reshape(1, -1))[0]
    return class_names[int(np.argmax(probs))], dict(zip(class_names, probs))

# --------------------------------------------------------------
# Paths & data
# --------------------------------------------------------------
BASE = Path(__file__).parent
MODEL_PATH  = BASE / "xgb_best_model.pkl"
ENC_PATH    = BASE / "best_model.pkl"
FEAT_PATH   = BASE / "Features.csv"
TARGET_PATH = BASE / "Target.csv"
NASDAQ_PATH = BASE / "Data" / "NASDAQ100.csv"

if "xgb_model" not in st.session_state:
    st.session_state["xgb_model"], st.session_state["class_names"] = load_model(MODEL_PATH, ENC_PATH)
model = st.session_state["xgb_model"]
class_names = st.session_state["class_names"]
features_df = load_csv(FEAT_PATH)

try:
    target_df = load_csv(TARGET_PATH)
except FileNotFoundError:
    target_df = None
nasdaq_df = load_csv(NASDAQ_PATH)

if not {"Open", "High", "Low", "Close"}.issubset(nasdaq_df.columns):
    nasdaq_df = nasdaq_df.assign(High=nasdaq_df["Close"], Low=nasdaq_df["Close"],
                                 Open=nasdaq_df["Open"].fillna(nasdaq_df["Close"]))

# --------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------
st.set_page_config(page_title="Global Echo - Market Predictor", layout="wide")
st.title("🌍 Global Echo - Market Predictor")
st.markdown("This app uses global market data to guess whether the NASDAQ will start the day up, down, or flat.")

st.markdown("""
---
#### ⚠️ Disclaimer
This application uses a machine learning model to estimate the likelihood of NASDAQ opening movements based on past global market data. It is a demonstration of AI capabilities for educational purposes only.  
**Do not use these predictions to make real financial decisions.**
""")

st.sidebar.success("Model: XGBoost (June 2025)")
st.sidebar.caption(f"Training span: {features_df.index.min().date()} → {features_df.index.max().date()}")

st.sidebar.header("Choose an input")
mode = st.sidebar.radio("Input mode", ["Pick historical date", "Enter my own data"], index=0)

if mode == "Pick historical date":
    available_dates = features_df.index.date.tolist()
    default_date = max(available_dates)
    selected_date = st.sidebar.date_input("Pick a date", default_date, min_value=min(available_dates), max_value=max(available_dates))
    if selected_date not in available_dates:
        st.warning("No data available for that date.")
        st.stop()
    row = features_df.loc[pd.to_datetime(selected_date)]
    pick = pd.to_datetime(selected_date)
    pick_str = pick.strftime("%Y-%m-%d")
    st.write(f"### Market data snapshot for **{pick_str}**")
    st.dataframe(row.to_frame().T, use_container_width=True)
else:
    with st.sidebar.expander("Manual Feature Entry"):
        manual_inputs = {col: st.number_input(col, value=0.0, step=0.1, help="Enter % change or signal value") for col in features_df.columns}
    row = pd.Series(manual_inputs)
    pick = None
    pick_str = "Manual entry"

st.divider()

# --------------------------------------------------------------
# Prediction & visualisation
# --------------------------------------------------------------
if st.button("Predict NASDAQ Opening", use_container_width=True):
    label, probs = predict_row(model, class_names, row)

    color_map = {"Up": "#2ecc71", "Down": "#e74c3c", "Flat": "#95a5a6"}
    emoji_map = {"Up": "📈", "Down": "📉", "Flat": "😐"}

    st.markdown("#### 📊 Prediction Confidence")
    fig_prob  = go.Figure()
    for cls in class_names:
        pct = probs[cls] * 100
        fig_prob.add_trace(go.Bar(x=[pct], y=[cls], orientation="h",
                                  marker_color=color_map.get(cls),
                                  text=f"{pct:.1f}%", textposition="inside",
                                  showlegend=False))
    fig_prob.update_layout(title=f"Predicted probabilities — NASDAQ gap on {pick_str}",
                           xaxis_title="Probability (%)", yaxis_title="",
                           height=260, margin=dict(l=60, r=40, t=50, b=30))
    st.plotly_chart(fig_prob, use_container_width=True)

    st.markdown(f"<h3 style='text-align:center'>Prediction: {emoji_map.get(label)} <span style='color:{color_map.get(label)}'>{label}</span> gap</h3>", unsafe_allow_html=True)

    if pick is not None and target_df is not None and pick in target_df.index:
        actual = target_df.loc[pick].squeeze()
        if actual == label:
            st.success(f"Model was **correct** - actual gap = {actual}")
        else:
            st.error(f"Model was **wrong** - actual gap = {actual}")

        all_bdays = pd.date_range(nasdaq_df.index.min(), nasdaq_df.index.max(), freq="B")
        missing   = all_bdays.difference(nasdaq_df.index)

        idx_pick  = nasdaq_df.index.get_indexer([pick], method="pad")[0]
        start_idx = max(0, idx_pick - 60)
        sub_df    = nasdaq_df.iloc[start_idx: idx_pick + 2]
        y_min, y_max = sub_df["Low"].min(), sub_df["High"].max()
        y_pad = (y_max - y_min) * 0.05

        vline_color = color_map.get(label, "red")

        st.markdown("#### 📅 Historical Candlestick Chart")
        fig_c = go.Figure()
        fig_c.add_trace(go.Candlestick(x=nasdaq_df.index,
                                        open=nasdaq_df["Open"], high=nasdaq_df["High"],
                                        low=nasdaq_df["Low"], close=nasdaq_df["Close"],
                                        name="NASDAQ Price"))
        fig_c.add_trace(go.Scatter(x=[pick, pick], y=[y_min, y_max], mode="lines",
                                   line=dict(color=vline_color, dash="dot"),
                                   name="Prediction day"))

        fig_c.update_xaxes(
            range=[sub_df.index[0], sub_df.index[-1]],
            rangebreaks=[dict(bounds=["sat", "mon"]), dict(values=missing)],
            rangeslider_visible=True
        )
        fig_c.update_yaxes(range=[y_min - y_pad, y_max + y_pad], fixedrange=False)
        fig_c.update_layout(height=650, margin=dict(l=30, r=30, t=30, b=10), showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_c, use_container_width=True)

st.markdown("---")
st.caption("Model: XGBoost | Features: Global % moves & shock signals | Data through Feb 2025")

with st.expander("ℹ️ How does this model work?"):
    st.markdown("""
    - We gather global market movement data.
    - Features include % changes in Europe, Asia, commodities, and economic signals.
    - The model learns from historical data to predict whether NASDAQ will open up, down, or flat.
    """)
with st.expander("ℹ️ How to use this app?"):
    st.markdown("""
    - **Pick a date**: Choose a historical date to see market data and predictions.
    - **Enter your own data**: Input your own % changes and signals to see how the model predicts.
    - Click **Predict NASDAQ Opening** to see the model's prediction and confidence.
    """)            