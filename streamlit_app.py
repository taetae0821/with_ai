import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from datetime import datetime, timezone, timedelta
import plotly.express as px

st.set_page_config(page_title="폭염 대시보드", layout="wide")
st.title("🌡️ 폭염 — 공개 데이터 시계열")
st.caption("NASA GISTEMP 공개 데이터를 사용하며, 연결 실패 시 예시 데이터를 사용합니다.")

# 오늘 날짜 (Asia/Seoul 기준)
def local_midnight_today():
    tz_offset = 9
    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    local_now = now_utc + timedelta(hours=tz_offset)
    local_midnight = datetime(year=local_now.year, month=local_now.month, day=local_now.day)
    return local_midnight - timedelta(hours=tz_offset)
LOCAL_MIDNIGHT_UTC = local_midnight_today()

# 공개 데이터 로드
GISTEMP_CSV_URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"

@st.cache_data(ttl=60*60*6)
def load_gistemp(url=GISTEMP_CSV_URL, timeout=10):
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        text = resp.text
        df = pd.read_csv(io.StringIO(text), skiprows=1)
        if 'Year' not in df.columns:
            df = df.rename(columns={df.columns[0]: 'Year'})

        month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        available_months = [m for m in month_names if m in df.columns]

        if available_months:
            df_melt = df.melt(
                id_vars=['Year'],
                value_vars=available_months,
                var_name='month',
                value_name='anom'
            )
            month_num = {m:i+1 for i,m in enumerate(month_names)}
            df_melt['month_num'] = df_melt['month'].map(month_num)
            df_melt['date'] = pd.to_datetime(df_melt['Year'].astype(str) + '-' + df_melt['month_num'].astype(str) + '-01')
            df_melt['anom'] = pd.to_numeric(df_melt['anom'].astype(str).str.replace('*',''), errors='coerce')
            df_final = df_melt[['date','anom']].rename(columns={'anom':'value'})
        else:
            df2 = df[['Year','J-D']].copy()
            df2['date'] = pd.to_datetime(df2['Year'].astype(str) + '-01-01')
            df2['value'] = pd.to_numeric(df2['J-D'], errors='coerce')
            df_final = df2[['date','value']].copy()

        df_final = df_final.drop_duplicates(subset=['date'])
        df_final = df_final[df_final['date'] < LOCAL_MIDNIGHT_UTC]
        return {"ok": True, "df": df_final}
    except Exception as e:
        # 예시 데이터
        example_dates = pd.date_range(end=(LOCAL_MIDNIGHT_UTC - pd.Timedelta(days=1)), periods=60, freq='M')
        ex_df = pd.DataFrame({
            'date': example_dates,
            'value': np.linspace(0.2, 1.2, len(example_dates)) + np.random.normal(scale=0.05, size=len(example_dates))
        })
        return {"ok": False, "df": ex_df, "error": str(e)}

load_result = load_gistemp()
if not load_result["ok"]:
    st.warning("공개 데이터 다운로드 실패 → 예시 데이터 사용\n오류: " + load_result.get("error","알 수 없음"))

df_plot = load_result["df"]

# 그래프 옵션
st.subheader("NASA GISTEMP — 월별 기온 이상값 시계열")
col1, col2 = st.columns([3,1])
with col2:
    rolling = st.selectbox("스무딩(개월)", [1,3,6,12], index=1)
    viz_type = st.selectbox("그래프 유형", ["꺾은선","면적"], index=0)
with col1:
    df_plot_vis = df_plot.copy()
    if rolling > 1:
        df_plot_vis['value_sm'] = df_plot_vis['value'].rolling(window=rolling, min_periods=1).mean()
        y_col = 'value_sm'
    else:
        y_col = 'value'

    # 기본 월별 표시
    x_col = 'date'

    if viz_type=="꺾은선":
        fig = px.line(df_plot_vis, x=x_col, y=y_col,
                      labels={x_col:'날짜', y_col:'이상값(°C)'})
    else:
        fig = px.area(df_plot_vis, x=x_col, y=y_col,
                      labels={x_col:'날짜', y_col:'이상값(°C)'})

    st.plotly_chart(fig, use_container_width=True)
