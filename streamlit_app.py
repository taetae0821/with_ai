"""
Streamlit 폭염 데이터 대시보드
- 공식 공개 데이터: NASA GISTEMP (글로벌 기온 이상값 CSV)
  URL: https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv
- 사용자 입력 글 제거, 폭염 중심 시각화
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from datetime import datetime, timezone, timedelta
import plotly.express as px

st.set_page_config(page_title="폭염 데이터 대시보드", layout="wide")

# Pretendard 폰트 적용 시도
PRETENDARD_PATH = "/fonts/Pretendard-Bold.ttf"
css_font = f"""
<style>
@font-face {{
  font-family: 'PretendardCustom';
  src: url('{PRETENDARD_PATH}');
}}
html, body, [class*="css"]  {{
  font-family: PretendardCustom, system-ui, -apple-system, "Segoe UI", Roboto, "Noto Sans KR", "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
}}
</style>
"""
st.markdown(css_font, unsafe_allow_html=True)

st.title("🌡️ 폭염 데이터 대시보드")

# 오늘(로컬 자정) 계산 (Asia/Seoul)
def local_midnight_today():
    tz_offset = 9
    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    local_now = now_utc + timedelta(hours=tz_offset)
    local_midnight = datetime(year=local_now.year, month=local_now.month, day=local_now.day)
    return local_midnight - timedelta(hours=tz_offset)

LOCAL_MIDNIGHT_UTC = local_midnight_today()

# ---------------------------
# NASA GISTEMP 데이터 로드
# ---------------------------
GISTEMP_CSV_URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"

@st.cache_data(ttl=60*60*6, show_spinner=False)
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
            df_final['group'] = 'GISTEMP월별'
        else:
            df2 = df[['Year','J-D']].copy()
            df2['date'] = pd.to_datetime(df2['Year'].astype(str) + '-01-01')
            df2['value'] = pd.to_numeric(df2['J-D'], errors='coerce')
            df_final = df2[['date','value']].copy()
            df_final['group'] = 'GISTEMP연간'

        df_final = df_final.drop_duplicates(subset=['date'])
        df_final = df_final[df_final['date'] < LOCAL_MIDNIGHT_UTC]
        return {"ok": True, "df": df_final, "source": url}

    except Exception as e:
        example_dates = pd.date_range(end=(LOCAL_MIDNIGHT_UTC - pd.Timedelta(days=1)), periods=60, freq='M')
        ex_df = pd.DataFrame({
            'date': example_dates,
            'value': np.linspace(0.2, 1.2, len(example_dates)) + np.random.normal(scale=0.05, size=len(example_dates)),
            'group': '예시_GISTEMP'
        })
        return {"ok": False, "df": ex_df, "error": str(e), "source": url}

# ---------------------------
# 공개 데이터 UI
# ---------------------------
load_result = load_gistemp()
if not load_result["ok"]:
    st.warning("공개 데이터 다운로드 실패 → 예시 데이터 사용\n오류: " + load_result.get("error","알 수 없음"))
gistemp_df = load_result["df"]

st.subheader("NASA GISTEMP — 기온 이상값 시계열")

# 연도/개월 선택
years = sorted(gistemp_df['date'].dt.year.unique())
selected_years = st.multiselect("연도 선택", years, default=[years[-5], years[-4], years[-3], years[-2], years[-1]] if len(years)>=5 else years)

df_plot = gistemp_df[gistemp_df['date'].dt.year.isin(selected_years)].copy()

rolling = st.selectbox("스무딩(개월)", [1,3,6,12], index=1)
viz_type = st.selectbox("그래프 유형", ["꺾은선","면적"], index=0)

if rolling > 1:
    df_plot['value_sm'] = df_plot['value'].rolling(window=rolling, min_periods=1).mean()
    y_col = 'value_sm'
else:
    y_col = 'value'

# 날짜별 평균 (중복 월 제거)
df_plot = df_plot.groupby('date', as_index=False)[y_col].mean()

if viz_type=="꺾은선":
    fig = px.line(df_plot, x='date', y=y_col, labels={'date':'날짜','value':'이상값(°C)'})
else:
    fig = px.area(df_plot, x='date', y=y_col, labels={'date':'날짜','value':'이상값(°C)'})

st.plotly_chart(fig, use_container_width=True)

st.download_button("CSV 다운로드", gistemp_df.to_csv(index=False).encode('utf-8'),
                   file_name="gistemp_preprocessed.csv", mime="text/csv")
