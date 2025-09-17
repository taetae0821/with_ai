import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.title("폭염 추이 대시보드")

# ----------------- 데이터 생성 -----------------
# 2015년~2025년 월별 가상의 평균 기온 데이터
dates = pd.date_range(start="2015-01-01", end="2025-12-31", freq='M')
temperatures = 25 + (dates.year - 2015) * 0.3 + np.random.randn(len(dates))  # 점점 높아지는 기온
df = pd.DataFrame({"date": dates, "temperature": temperatures})

# ----------------- 사이드바: 연도 선택 -----------------
years = df['date'].dt.year.unique()
selected_years = st.sidebar.multiselect("연도 선택", options=years, default=years)

# 선택한 연도만 필터링
df_plot = df[df['date'].dt.year.isin(selected_years)].copy()

# ----------------- 사이드바: 표시 단위 선택 -----------------
unit = st.sidebar.radio("표시 단위", options=["월별", "연별"])

if unit == "연별":
    df_plot = df_plot.groupby(df_plot['date'].dt.year, as_index=False).mean()
    df_plot.rename(columns={'date': 'year'}, inplace=True)
    x_col = "year"
    y_col = "temperature"
else:
    x_col = "date"
    y_col = "temperature"

# ----------------- 꺾은선 그래프 -----------------
fig = px.line(df_plot, x=x_col, y=y_col, markers=True,
              title="폭염 추이", labels={x_col: x_col, y_col: "평균 기온(°C)"})

st.plotly_chart(fig, use_container_width=True)
