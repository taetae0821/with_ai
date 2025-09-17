"""
Streamlit 대시보드 (한국어 UI)
- 공개 데이터: NASA GISTEMP (글로벌 기온 이상값 CSV)
- 사용자 입력 글 제거
- plotly 없이 matplotlib + seaborn 사용
- 월별/연도별 선택 가능
- Pretendard 폰트 적용 시도
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# 페이지 설정
# ---------------------------
st.set_page_config(page_title="폭염 대시보드", layout="wide")

# Pretendard 폰트 시도
PRETENDARD_PATH = "/fonts/Pretendard-Bold.ttf"
css_font = f"""
<style>
@font-face {{
  font-family: 'PretendardCustom';
  src: url('{PRETENDARD_PATH}');
}}
html, body, [class*="css"] {{
  font-family: PretendardCustom, system-ui, -apple-system, "Segoe UI", Roboto, "Noto Sans KR", "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
}}
</style>
"""
st.markdown(css_font, unsafe_allow_html=True)

st.title("🌡️ 폭염 대시보드 — 공개 데이터 분석")

# ---------------------------
# 로컬 자정 계산 (Asia/Seoul)
# ---------------------------
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
            df_melt['value'] = pd.to_numeric(df_melt['anom'].astype(str).str.replace('*',''), errors='coerce')
            df_final = df_melt[['date','value']].copy()
        else:
            df2 = df[['Year','J-D']].copy()
            df2['date'] = pd.to_datetime(df2['Year'].astype(str) + '-01-01')
            df2['value'] = pd.to_numeric(df2['J-D'], errors='coerce')
            df_final = df2[['date','value']].copy()

        df_final = df_final.drop_duplicates(subset=['date'])
        df_final = df_final[df_final['date'] < LOCAL_MIDNIGHT_UTC].reset_index(drop=True)
        return {"ok": True, "df": df_final, "source": url}
    except Exception as e:
        example_dates = pd.date_range(end=(LOCAL_MIDNIGHT_UTC - pd.Timedelta(days=1)), periods=60, freq='M')
        ex_df = pd.DataFrame({
            'date': example_dates,
            'value': np.linspace(0.2, 1.2, len(example_dates)) + np.random.normal(scale=0.05, size=len(example_dates)),
        })
        return {"ok": False, "df": ex_df, "error": str(e), "source": url}

# ---------------------------
# 데이터 로드
# ---------------------------
load_result = load_gistemp()
if not load_result["ok"]:
    st.warning("공개 데이터(GISTEMP) 다운로드 실패 → 예시 데이터 사용\n오류: " + load_result.get("error", "알 수 없음"))
gistemp_df = load_result["df"]

# 상단 요약
col1, col2, col3 = st.columns([2,2,2])
with col1:
    st.metric("데이터 시작", gistemp_df['date'].min().strftime("%Y-%m-%d"))
with col2:
    st.metric("데이터 끝", gistemp_df['date'].max().strftime("%Y-%m-%d"))
with col3:
    st.metric("샘플 수", f"{len(gistemp_df):,}")

# ---------------------------
# 시각화 옵션
# ---------------------------
st.subheader("기온 이상값 시계열")
colA, colB = st.columns([3,1])
with colB:
    rolling = st.selectbox("스무딩(개월)", [1,3,6,12], index=1)
    view_type = st.selectbox("표시 단위", ["월별","연별"], index=0)
with colA:
    df_plot = gistemp_df.copy()
    if view_type=="연별":
        df_plot = df_plot.groupby(df_plot['date'].dt.year).mean().reset_index()
        df_plot['date'] = pd.to_datetime(df_plot['date'], format='%Y')
    if rolling > 1:
        df_plot['value'] = df_plot['value'].rolling(window=rolling, min_periods=1).mean()

    # matplotlib + seaborn
    plt.figure(figsize=(12,5))
    sns.lineplot(data=df_plot, x='date', y='value', marker='o')
    plt.xlabel("날짜")
    plt.ylabel("이상값(°C)")
    plt.title("전지구 표면 온도 이상값 (NASA GISTEMP)")
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())
    plt.close()

# ---------------------------
# CSV 다운로드
# ---------------------------
st.download_button("CSV 다운로드", gistemp_df.to_csv(index=False).encode('utf-8'),
                   file_name="gistemp_preprocessed.csv", mime="text/csv")

# ---------------------------
# 마무리
# ---------------------------
st.markdown("---")
st.subheader("참고")
st.markdown("""
- NASA GISS GISTEMP (GLB.Ts+dSST.csv) — Global Land-Ocean Temperature Index  
  URL: https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv
""")
