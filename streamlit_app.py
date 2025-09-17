# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="폭염 대시보드", layout="wide")

st.title("📈 폭염 연도별 대시보드")

# ----- 데이터 생성 -----
# 2015~2025년 월별 날짜
dates = pd.date_range(start="2015-01-01", end="2025-12-31", freq='M')

# 날짜가 datetime이 아니면 변환
dates = pd.to_datetime(dates)

# 온도 생성: 연도 증가에 따라 조금씩 상승 + 랜덤 노이즈
temperatures = 25 + (dates.year - 2015) * 0.3 + np.random.randn(len(dates))

df = pd.DataFrame({
    'date': dates,
    'temperature': temperatures
})

# ----- 사용자 선택 -----
st.sidebar.header("연도 선택")
years = df['date'].dt.year.unique()
selected_years = st.sidebar.multiselect("연도를 선택하세요", options=years, default=years)

# 선택한 연도 데이터만 필터링
df_plot = df[df['date'].dt.year.isin(selected_years)].copy()

# 연별 평균 온도 계산
df_plot = df_plot.groupby(df_plot['date'].dt.year).mean().reset_index()
df_plot.rename(columns={'date':'year'}, inplace=True)

# ----- 그래프 -----
fig = px.bar(df_plot, x='year', y='temperature',
             labels={'year':'연도', 'temperature':'평균 온도 (℃)'},
             title='연도별 평균 폭염 온도')
st.plotly_chart(fig, use_container_width=True)

# ----- 간단 설명 -----
st.markdown("---")
st.subheader("📌 폭염 현황 요약")
st.write(
    "최근 몇 년간 여름이 점점 길어지고 폭염일수가 증가하고 있습니다. "
    "교실 환경과 야외 활동 시 주의가 필요하며, 적절한 대응이 필요합니다."
)
