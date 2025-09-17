# streamlit_app.py
# Streamlit 앱 — 한국어 UI
# 주요 공개 데이터(예시): NOAA / NASA / World Bank 등
# 출처:
# NOAA Global Temperature anomalies: https://www.ncei.noaa.gov/access/monitoring/global-temperature-anomalies/
# NASA GISTEMP: https://data.giss.nasa.gov/gistemp/
# World Bank CO2 (kt): http://api.worldbank.org/v2/en/indicator/EN.ATM.CO2E.KT?downloadformat=csv

import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime
import time
import os

st.set_page_config(page_title="데이터 대시보드 (Streamlit + Codespaces)", layout="wide")

# -----------------------
# 유틸리티
# -----------------------
def drop_future_dates(df, date_col='date'):
    if date_col not in df.columns:
        return df
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    cutoff = pd.to_datetime(datetime.utcnow())
    return df[df[date_col] <= cutoff].copy()

# -----------------------
# 폰트 적용 시도 (Pretendard)
# -----------------------
PRETENDARD_PATH = "/fonts/Pretendard-Bold.ttf"
def try_apply_pretendard():
    try:
        if os.path.exists(PRETENDARD_PATH):
            import matplotlib.font_manager as fm
            fm.fontManager.addfont(PRETENDARD_PATH)
            prop = fm.FontProperties(fname=PRETENDARD_PATH)
            plt.rcParams['font.family'] = prop.get_name()
        if os.path.exists(PRETENDARD_PATH):
            st.markdown(f"""
            <style>
            @font-face {{
                font-family: 'PretendardCustom';
                src: url('{PRETENDARD_PATH}') format('truetype');
            }}
            html, body, .css-1d391kg, .stApp {{
                font-family: PretendardCustom, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans KR', 'Helvetica', 'Arial', sans-serif;
            }}
            </style>
            """, unsafe_allow_html=True)
    except Exception:
        pass

try_apply_pretendard()

# -----------------------
# 캐시 및 재시도 유틸리티
# -----------------------
MAX_RETRIES = 3
RETRY_DELAY = 1.0

@st.cache_data(show_spinner=False)
def fetch_with_retries(url, params=None, headers=None, as_bytes=False, timeout=15):
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.content if as_bytes else resp.text
        except Exception as e:
            last_exc = e
            time.sleep(RETRY_DELAY * attempt)
    raise last_exc

# -----------------------
# 공개 데이터 불러오기
# -----------------------
def load_public_noaa():
    url = "https://www.ncei.noaa.gov/access/monitoring/global-temperature-anomalies/monthly.csv"
    try:
        txt = fetch_with_retries(url)
        df = pd.read_csv(StringIO(txt))
        if 'Date' in df.columns:
            df = df.rename(columns={'Date':'date'})
        if 'Value' in df.columns:
            df = df.rename(columns={'Value':'value'})
        df = df[['date','value']].copy()
        df = df.dropna(subset=['date'])
        df = drop_future_dates(df, 'date')
        return df, None
    except Exception:
        dates = pd.date_range(end=pd.Timestamp.utcnow(), periods=240, freq='M')
        anomalies = np.cumsum(np.random.normal(0.01, 0.05, size=len(dates)))
        df = pd.DataFrame({'date':dates, 'value':anomalies})
        msg = "공개 데이터(예: NOAA) 불러오기에 실패하여 예시 데이터를 사용합니다."
        return df, msg

def load_public_co2_worldbank():
    url = "http://api.worldbank.org/v2/country/all/indicator/EN.ATM.CO2E.KT?format=json&per_page=10000"
    try:
        txt = fetch_with_retries(url)
        data = pd.read_json(StringIO(txt))
        records = pd.DataFrame(data[1])
        df = records[['date','value']].copy()
        df['date'] = pd.to_datetime(df['date'], format='%Y', errors='coerce')
        df = drop_future_dates(df, 'date')
        return df, None
    except Exception:
        years = np.arange(1980, pd.Timestamp.utcnow().year+1)
        vals = np.linspace(1000000, 1500000, len(years)) + np.random.normal(0, 50000, len(years))
        df = pd.DataFrame({'date':pd.to_datetime(years, format='%Y'), 'value':vals})
        msg = "공개 데이터(예: World Bank CO2) 불러오기에 실패하여 예시 데이터를 사용합니다."
        return df, msg

# -----------------------
# 공개 데이터 대시보드
# -----------------------
def public_dashboard():
    st.header("공식 공개 데이터 대시보드")
    df_temp, msg_temp = load_public_noaa()
    if msg_temp:
        st.warning(msg_temp)
    st.write(df_temp.head(10))

    df_co2, msg_co2 = load_public_co2_worldbank()
    if msg_co2:
        st.warning(msg_co2)
    st.write(df_co2.head(10))

# -----------------------
# 사용자 입력 대시보드 (예시 데이터만)
# -----------------------
def user_input_dashboard():
    st.header("사용자 입력 기반 대시보드")
    st.info("현재 입력된 CSV/이미지/설명이 없어 예시 데이터를 사용합니다.")
    dates = pd.date_range(end=pd.Timestamp.utcnow(), periods=36, freq='M')
    groups = ['A','B','C']
    rows = []
    for g in groups:
        vals = np.cumsum(np.random.normal(10,5,len(dates))) + (0 if g=='A' else 50 if g=='B' else 100)
        for d,v in zip(dates, vals):
            rows.append({'date':d, 'value':v, 'group':g})
    df_user = pd.DataFrame(rows)
    st.write(df_user.head(10))

# -----------------------
# 메인
# -----------------------
def main():
    st.title("Streamlit + GitHub Codespaces 데이터 대시보드")
    st.sidebar.title("네비게이션")
    page = st.sidebar.radio("페이지 선택", ["공개 데이터 대시보드", "사용자 입력 대시보드", "앱 정보"])
    if page == "공개 데이터 대시보드":
        public_dashboard()
    elif page == "사용자 입력 대시보드":
        user_input_dashboard()
    else:
        st.header("앱 정보")
        st.markdown("간단한 Streamlit 대시보드 예시입니다.")

if __name__ == "__main__":
    main()
