# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="폭염 대시보드", layout="wide")

st.title("📈 폭염 연도별 분석 대시보드")

# ----- 데이터 생성 -----
dates = pd.date_range(start="2015-01-01", end="2025-08-01", freq="M")
temperatures = 25 + (dates.year - 2015) * 0.3 + np.random.randn(len(dates))
df = pd.DataFrame({"date": dates, "temperature": temperatures})

# ----- 연도 선택 -----
years = df['date'].dt.year.unique()
selected_years = st.multiselect("연도 선택", options=years, default=years.tolist())

# 선택된 연도 필터링
df_filtered = df[df['date'].dt.year.isin(selected_years)]

# ----- 연별 평균 계산 -----
df_plot = df_filtered.groupby(df_filtered['date'].dt.year, as_index=False).mean()
df_plot.rename(columns={'date': 'year'}, inplace=True)

# ----- 그래프 -----
fig = px.bar(df_plot, x='year', y='temperature',
             labels={'year': '연도', 'temperature': '평균 기온 (℃)'},
             title="연도별 평균 기온 추이")

st.plotly_chart(fig, use_container_width=True)

# ----- 추가 설명 -----
st.markdown("""
최근 몇 년간 여름이 길어지고, 폭염일수가 증가하고 있습니다.
교실 환경과 건강을 위해 적절한 대응이 필요합니다.
""")
