import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# 🎨 Page config and custom CSS
st.set_page_config(page_title="Stock Dashboard", layout="wide")
st.markdown("""
    <style>
    .main {background-color: #f9f9f9;}
    h1, h2, h3 {color: #2e8b57;}
    footer {visibility: hidden;}
    .stButton>button {background-color: #2e8b57; color: white;}
    </style>
""", unsafe_allow_html=True)

# 🏷️ Title
st.title("📈 Real-Time Stock Price Dashboard")

# 🎛️ Sidebar filters
st.sidebar.header("Filter Options")
ticker = st.sidebar.text_input("Stock Symbol", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# 📥 Fetch data
with st.spinner('Fetching Data...'):
    data = yf.download(ticker, start=start_date, end=end_date)
    # Flatten multi-index columns
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    st.write(f"Data shape: {data.shape}")

if st.sidebar.button("🔄 Retry Fetch"):
    data = yf.download(ticker, start=start_date, end=end_date)

# 📊 Display data
if not data.empty:
    st.subheader(f"{ticker} Stock Data")
    st.dataframe(data.tail(), use_container_width=True)

    # 📈 Interactive chart
    fig = px.line(data, x=data.index, y='Close', title=f"{ticker} Closing Prices")
    st.plotly_chart(fig, use_container_width=True)

    # 📥 Download button
    st.download_button("📥 Download CSV", data.to_csv(), "stock_data.csv")

    # 📉 Prediction
    if len(data) > 30:
        st.subheader("📉 Predict Next Day's Price")

        data['Days'] = np.arange(len(data)).reshape(-1, 1)
        model = LinearRegression()
        model.fit(data[['Days']], data['Close'])

        next_day = np.array([[len(data)]])
        prediction = model.predict(next_day)[0]

        # 📐 Confidence interval
        error_margin = 1.25
        lower = prediction - error_margin
        upper = prediction + error_margin

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Close", f"${prediction:.2f}")
        with col2:
            st.write(f"Range: ${lower:.2f} – ${upper:.2f}")

else:
    st.warning("⚠️ No data found. Please check the stock symbol.")

# 🪄 Footer
st.markdown("---")
st.markdown("Made with ❤️ by Dinesh K | Powered by Streamlit & yfinance")
