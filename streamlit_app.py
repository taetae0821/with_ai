import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="폭염 대시보드", layout="wide")
st.title("폭염 대시보드")

# ----- 예시 데이터 생성 -----
# 실제로는 CSV 파일이나 API에서 데이터를 불러오세요
dates = pd.date_range(start="2015-01-01", end="2025-08-01", freq='M')
temperatures = 25 + (dates.year - 2015) * 0.3 + pd.Series(pd.np.random.randn(len(dates)))
df = pd.DataFrame({"date": dates, "temperature": temperatures})

# ----- 연도 선택 -----
years = df['date'].dt.year.unique()
selected_year = st.selectbox("연도를 선택하세요", years)

# 선택 연도 데이터만
df_selected = df[df['date'].dt.year == selected_year]

# ----- 연별 평균 -----
df_yearly = df.groupby(df['date'].dt.year, as_index=False).mean()
df_yearly = df_yearly.rename(columns={'date': 'year'})

# ----- 그래프 표시 -----
st.subheader("선택 연도 월별 폭염 추이")
fig_month = px.line(df_selected, x='date', y='temperature',
                    title=f"{selected_year}년 월별 평균 기온",
                    labels={"date": "월", "temperature": "평균 기온(℃)"})
st.plotly_chart(fig_month, use_container_width=True)

st.subheader("연도별 평균 폭염 추이")
fig_year = px.bar(df_yearly, x='year', y='temperature',
                  title="연도별 평균 기온",
                  labels={"year": "연도", "temperature": "평균 기온(℃)"})
st.plotly_chart(fig_year, use_container_width=True)
