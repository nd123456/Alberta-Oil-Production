import streamlit as st
import pandas as pd
import plotly.express as px
import snowflake.connector
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Energy Analytics", layout="wide")

st.title("Alberta Oil Production Analytics")

# Snowflake connection
conn = snowflake.connector.connect(
    user="USER",
    password="PASSWORD",
    account="ACCOUNTID",
    warehouse="COMPUTE_WH",
    database="ENERGY_ANALYTICS",
    schema="ENERGY_SCHEMA"
)

query = """
SELECT d.DATE, o.OIL_TYPE, f.PRODUCTION_VOLUME
FROM FACT_OIL_PRODUCTION f
JOIN DIM_DATE d ON f.DATE = d.DATE
JOIN DIM_OIL_TYPE o ON f.TYPE_ID = o.TYPE_ID
"""

df = pd.read_sql(query, conn)

df["DATE"] = pd.to_datetime(df["DATE"])
df["YEAR"] = df["DATE"].dt.year

# Sidebar filters
st.sidebar.header("Filters")

oil_types = df["OIL_TYPE"].unique().tolist()

oil_type = st.sidebar.multiselect(
    "Oil Type",
    options=oil_types,
    default=oil_types
)

year_range = st.sidebar.slider(
    "Year Range",
    int(df["YEAR"].min()),
    int(df["YEAR"].max()),
    (int(df["YEAR"].min()), int(df["YEAR"].max()))
)

filtered = df[
    (df["OIL_TYPE"].isin(oil_type)) &
    (df["YEAR"] >= year_range[0]) &
    (df["YEAR"] <= year_range[1])
]

# KPIs
total_prod = filtered["PRODUCTION_VOLUME"].sum()
avg_prod = filtered["PRODUCTION_VOLUME"].mean()

col1, col2 = st.columns(2)

col1.metric("Total Production", f"{total_prod:,.0f}")
col2.metric("Average Production", f"{avg_prod:,.0f}")

# Production trend chart
filtered = filtered.sort_values("DATE")

trend = (
    filtered
    .groupby("DATE", as_index=False)["PRODUCTION_VOLUME"]
    .sum()
    .sort_values("DATE")
)

fig = px.line(
    trend,
    x="DATE",
    y="PRODUCTION_VOLUME",
    title="Production Trend",
)

st.plotly_chart(fig, use_container_width=True)

# Yearly production chart
yearly = filtered.groupby("YEAR")["PRODUCTION_VOLUME"].sum().reset_index()

fig2 = px.bar(
    yearly,
    x="YEAR",
    y="PRODUCTION_VOLUME",
    title="Yearly Production"
)

st.plotly_chart(fig2, use_container_width=True)

# Forecast section
st.subheader("Production Forecast")

ts = df.groupby("DATE")["PRODUCTION_VOLUME"].sum().reset_index()

ts["t"] = np.arange(len(ts))

model = LinearRegression()
model.fit(ts[["t"]], ts["PRODUCTION_VOLUME"])

future = pd.DataFrame({
    "t": np.arange(len(ts), len(ts)+24)
})

future["forecast"] = model.predict(future)

forecast_dates = pd.date_range(
    ts["DATE"].max(),
    periods=24,
    freq="M"
)

forecast_df = pd.DataFrame({
    "DATE": forecast_dates,
    "PRODUCTION_VOLUME": future["forecast"]
})

fig3 = px.line(ts, x="DATE", y="PRODUCTION_VOLUME", title="Production Forecast")

fig3.add_scatter(
    x=forecast_df["DATE"],
    y=forecast_df["PRODUCTION_VOLUME"],
    mode="lines",
    name="Forecast"
)

st.plotly_chart(fig3, use_container_width=True)

# Data explorer
st.subheader("Raw Data")

st.dataframe(filtered)