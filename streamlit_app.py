# streamlit_app.py
# Streamlit 앱 — 한국어 UI
# 주요 공개 데이터(예시): NOAA / NASA / World Bank 등
# 출처(예시, 코드 주석에 남김):
# NOAA Global Temperature anomalies: https://www.ncei.noaa.gov/access/monitoring/global-temperature-anomalies/
# NASA GISTEMP: https://data.giss.nasa.gov/gistemp/
# World Bank CO2 (kt): http://api.worldbank.org/v2/en/indicator/EN.ATM.CO2E.KT?downloadformat=csv
# (kaggle 사용 시) kaggle API 인증 안내: https://www.kaggle.com/docs/api

import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO, StringIO
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import time
import os

st.set_page_config(page_title="데이터 대시보드 (Streamlit + Codespaces)", layout="wide")

# -----------------------
# 유틸리티: 한국어 날짜/형식
# -----------------------
def now_seoul():
    # 로컬 타임존이 아닌 환경에서도 UTC를 기반으로 현재 시각을 얻은 후 한국 시간으로 변환
    return datetime.utcnow().astimezone().astimezone()

def drop_future_dates(df, date_col='date'):
    if date_col not in df.columns:
        return df
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        cutoff = pd.to_datetime(datetime.utcnow())
        return df[df[date_col] <= cutoff].copy()
    except Exception:
        return df

# -----------------------
# 폰트 적용 시도 (Pretendard)
# -----------------------
PRETENDARD_PATH = "/fonts/Pretendard-Bold.ttf"
def try_apply_pretendard():
    # matplotlib
    try:
        if os.path.exists(PRETENDARD_PATH):
            import matplotlib.font_manager as fm
            fm.fontManager.addfont(PRETENDARD_PATH)
            prop = fm.FontProperties(fname=PRETENDARD_PATH)
            plt.rcParams['font.family'] = prop.get_name()
    except Exception:
        pass

    # plotly: set default font family in layouts later per-figure
    # streamlit itself will inherit system fonts; can't guarantee Pretendard but attempt to reference it in CSS
    try:
        if os.path.exists(PRETENDARD_PATH):
            st.markdown(
                f"""
                <style>
                @font-face {{
                    font-family: 'PretendardCustom';
                    src: url('{PRETENDARD_PATH}') format('truetype');
                }}
                html, body, .css-1d391kg, .stApp {{
                    font-family: PretendardCustom, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans KR', 'Helvetica', 'Arial', sans-serif;
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )
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
    # 재시도 실패 시 예외 전달
    raise last_exc

# -----------------------
# 공개 데이터 불러오기 (예시: NOAA global temp anomalies + World Bank CO2)
# - 실패 시 예시 데이터로 대체하고 사용자에게 안내
# -----------------------
def load_public_noaa():
    """
    NOAA (예시)에서 월별 전지구 온도 편차 데이터를 시도해서 불러옵니다.
    (실제 서비스 URL은 변경될 수 있음 — 실행 환경에서 접속 실패하면 예시 데이터로 대체됩니다.)
    출처 예시:
      https://www.ncei.noaa.gov/access/monitoring/global-temperature-anomalies/
    """
    # 예시 URL (직접 CSV 링크가 바뀔 수 있으므로 실패 대비)
    # 여기서는 공개 데이터를 얻기 위한 시도용 URL을 명시합니다.
    url = "https://www.ncei.noaa.gov/access/monitoring/global-temperature-anomalies/monthly.csv"
    try:
        txt = fetch_with_retries(url, as_bytes=False)
        # 데이터 파싱 (예상 포맷: 연-월, anomaly)
        df = pd.read_csv(StringIO(txt))
        # 표준화: date, value, group(optional)
        # 사용자 데이터 형식이 다양할 수 있으므로 안전하게 변환
        if 'Date' in df.columns:
            df = df.rename(columns={'Date':'date'})
        if 'Value' in df.columns:
            df = df.rename(columns={'Value':'value'})
        # try to find a column with 'anomaly' or 'temp'
        for col in df.columns:
            if 'anom' in col.lower() or 'temp' in col.lower():
                df = df.rename(columns={col:'value'})
                break
        if 'date' not in df.columns:
            # try to combine Year & Month
            if {'Year','Month'}.issubset(df.columns):
                df['date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01', errors='coerce')
        df = df[['date','value']].copy()
        df = df.dropna(subset=['date'])
        df = drop_future_dates(df, 'date')
        return df, None
    except Exception as e:
        # 실패 시 예시 데이터 생성
        dates = pd.date_range(end=pd.Timestamp.utcnow(), periods=240, freq='M')
        np.random.seed(0)
        anomalies = np.cumsum(np.random.normal(0.01, 0.05, size=len(dates)))  # fake trend
        sample = pd.DataFrame({'date':dates, 'value':anomalies})
        msg = "공개 데이터(예: NOAA) 불러오기에 실패하여 예시 데이터를 사용합니다."
        return sample, msg

def load_public_co2_worldbank():
    """
    World Bank CO2 indicator 예시를 시도해서 불러옵니다.
    출처 예시:
      http://api.worldbank.org/v2/en/indicator/EN.ATM.CO2E.KT?downloadformat=csv
    """
    # World Bank 직접 API (CSV 다운로드가 zip일 수 있음). 여기서는 간편화.
    url = "http://api.worldbank.org/v2/country/all/indicator/EN.ATM.CO2E.KT?format=json&per_page=10000"
    try:
        txt = fetch_with_retries(url, as_bytes=False)
        data = pd.read_json(StringIO(txt))
        # World Bank JSON 구조: [metadata, [records...]]
        if isinstance(data, list) and len(data) >= 2:
            records = pd.DataFrame(data[1])
            # 표준화
            if 'date' in records.columns and 'value' in records.columns:
                df = records[['date','value']].copy()
                df['date'] = pd.to_datetime(df['date'], format='%Y', errors='coerce')
                df = drop_future_dates(df, 'date')
                return df, None
        raise ValueError("World Bank 포맷 예외")
    except Exception as e:
        years = np.arange(1980, pd.Timestamp.utcnow().year+1)
        vals = np.linspace(1000000, 1500000, len(years)) + np.random.normal(0, 50000, len(years))
        sample = pd.DataFrame({'date':pd.to_datetime(years, format='%Y'), 'value':vals})
        msg = "공개 데이터(예: World Bank CO2) 불러오기에 실패하여 예시 데이터를 사용합니다."
        return sample, msg

# -----------------------
# 공개 데이터 대시보드 렌더링
# -----------------------
def public_dashboard():
    st.header("공식 공개 데이터 대시보드")
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("전지구 평균 표면 온도 편차 (월별) — NOAA 예시")
        df_temp, msg_temp = load_public_noaa()
        if msg_temp:
            st.warning(msg_temp)
        df_temp = df_temp.sort_values('date')
        # 전처리: 결측/중복/형변환
        df_temp = df_temp.drop_duplicates()
        df_temp['value'] = pd.to_numeric(df_temp['value'], errors='coerce')
        df_temp = df_temp.dropna(subset=['value'])
        st.write("데이터 미리보기 (최대 10개)", df_temp.head(10))

        # 사이드바 옵션 자동 구성
        st.sidebar.markdown("### 공용 데이터 옵션")
        # 기간 필터
        min_date = df_temp['date'].min()
        max_date = df_temp['date'].max()
        start, end = st.sidebar.date_input("기간 필터 (공개 데이터: 온도)", [min_date.date(), max_date.date()])
        # 스무딩 옵션
        smoothing = st.sidebar.slider("스무딩 이동평균(개월)", min_value=1, max_value=24, value=3)
        # 단위 변환(온도 anomaly는 보통 °C)
        unit_label = "°C (Anomaly)"

        # 필터 적용
        mask = (df_temp['date'] >= pd.to_datetime(start)) & (df_temp['date'] <= pd.to_datetime(end))
        df_temp_plot = df_temp.loc[mask].copy()
        if smoothing > 1:
            df_temp_plot['smoothed'] = df_temp_plot['value'].rolling(window=smoothing, min_periods=1).mean()
            y = 'smoothed'
            legend = f"이동평균({smoothing}개월)"
        else:
            y = 'value'
            legend = "원본"

        fig = px.line(df_temp_plot, x='date', y=y, title=f"전지구 평균 표면 온도 편차 ({unit_label})",
                      labels={'date':'날짜','smoothed':'온도(이동평균)','value':'온도(원본)'},
                      template='plotly_white')
        fig.update_layout(font_family="PretendardCustom, 'Noto Sans KR', sans-serif")
        st.plotly_chart(fig, use_container_width=True)

        # CSV 다운로드
        csv_buf = df_temp_plot.to_csv(index=False).encode('utf-8')
        st.download_button("전처리된 공개 데이터 CSV 다운로드", data=csv_buf, file_name="public_temp_preprocessed.csv", mime="text/csv")

    with col2:
        st.subheader("연간 국가별 CO₂ 배출(예시: World Bank)")
        df_co2, msg_co2 = load_public_co2_worldbank()
        if msg_co2:
            st.warning(msg_co2)
        df_co2 = df_co2.drop_duplicates()
        df_co2['value'] = pd.to_numeric(df_co2['value'], errors='coerce')
        df_co2 = df_co2.dropna(subset=['value'])
        # 간단 시각화: 연도별 합계 (예시 데이터가 국가별이면 집계 처리 필요 — 여기선 이미 연단위 집계 예시로 가정)
        df_co2 = df_co2.sort_values('date')
        st.write("데이터 미리보기 (최대 10개)", df_co2.head(10))

        # 연도 선택
        years = df_co2['date'].dt.year.unique()
        sel_year = st.selectbox("연도 선택 (CO₂)", options=sorted(years)[-10:], index=len(sorted(years)) - 1)
        df_year = df_co2[df_co2['date'].dt.year == int(sel_year)]
        if df_year.empty:
            st.info("선택한 연도의 데이터가 없습니다.")
        else:
            fig2 = px.bar(df_year, x='date', y='value', title=f"{sel_year}년 CO₂ (단위: kt)",
                          labels={'date':'연도','value':'CO₂ (kt)'}, template='plotly_white')
            fig2.update_layout(font_family="PretendardCustom, 'Noto Sans KR', sans-serif")
            st.plotly_chart(fig2, use_container_width=True)
            csv_buf2 = df_year.to_csv(index=False).encode('utf-8')
            st.download_button("선택 연도 CO₂ CSV 다운로드", data=csv_buf2, file_name=f"co2_{sel_year}.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("**설명:** 먼저 공식 공개 데이터를 불러와 기본적인 전처리(결측, 형변환, 미래 데이터 제거)를 수행합니다. 데이터 소스가 변경되거나 접근 불가하면 예시 데이터로 대체됩니다.")

# -----------------------
# 사용자 입력 대시보드
# - 원칙: 입력 섹션에 제공된 자료만 사용. (현재 프롬프트로 받은 input 없음 -> 예시 알림 및 내장 샘플 사용)
# - 앱 실행 중 파일 업로드/텍스트 요구 금지
# -----------------------
def user_input_dashboard():
    st.header("사용자 입력 기반 대시보드")
    st.info("참고: 현재 입력된 CSV/이미지/설명이 없습니다. (프롬프트의 Input 섹션 미제공) 예시 데이터를 사용하여 대시보드를 자동 생성합니다.")
    # 여기는 '사용자가 제공한' 데이터만 사용해야 함. 입력이 없으므로 예시 데이터 사용 및 안내.
    # 예시: 간단한 지역별 판매/측정치 CSV 포맷 (date, value, group)
    dates = pd.date_range(end=pd.Timestamp.utcnow(), periods=36, freq='M')
    groups = ['A','B','C']
    rows = []
    for g in groups:
        vals = np.cumsum(np.random.normal(10, 5, size=len(dates))) + (0 if g=='A' else 50 if g=='B' else 100)
        for d,v in zip(dates, vals):
            rows.append({'date':d, 'value':v, 'group':g})
    df_user = pd.DataFrame(rows)
    # 표준화 및 전처리
    df_user = df_user.drop_duplicates()
    df_user['date'] = pd.to_datetime(df_user['date'], errors='coerce')
    df_user = drop_future_dates(df_user, 'date')
    df_user['value'] = pd.to_numeric(df_user['value'], errors='coerce')
    df_user = df_user.dropna(subset=['date','value'])
    st.write("사용자 입력(또는 예시) 데이터 미리보기", df_user.head(10))

    # 자동으로 시각화 선택: 시계열 + 그룹 -> 꺾은선(그룹별)
    st.subheader("시계열: 그룹별 추이")
    smoothing = st.slider("스무딩 이동평균(기간)", min_value=1, max_value=12, value=3, key='user_smooth')
    agg_df = df_user.sort_values('date').copy()
    if smoothing > 1:
        agg_df['value_sm'] = agg_df.groupby('group')['value'].transform(lambda x: x.rolling(window=smoothing, min_periods=1).mean())
        y_col = 'value_sm'
    else:
        y_col = 'value'
    fig = px.line(agg_df, x='date', y=y_col, color='group', title="그룹별 시계열 추이",
                  labels={'date':'날짜', y_col:'값'}, template='plotly_white')
    fig.update_layout(font_family="PretendardCustom, 'Noto Sans KR', sans-serif")
    st.plotly_chart(fig, use_container_width=True)

    # 비율 시각화: 최신 시점에서 그룹 비중 (원그래프)
    st.subheader("비율: 최신 시점 그룹 비중")
    latest_date = agg_df['date'].max()
    df_latest = agg_df[agg_df['date'] == latest_date].groupby('group', as_index=False)[y_col].sum()
    if not df_latest.empty:
        fig2 = px.pie(df_latest, names='group', values=y_col, title=f"{latest_date.date()} 기준 그룹 비중",
                      hole=0.35)
        fig2.update_layout(font_family="PretendardCustom, 'Noto Sans KR', sans-serif")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("최근 시점 데이터가 없어 비율 차트를 표시할 수 없습니다.")

    # 지도 시각화: 만약 group을 지역코드로 쓰면 지도 표시 (예시로 그룹->위치 매핑)
    st.subheader("지역(그룹) 지도 시각화 (예시)")
    # 예시 좌표 매핑
    loc_map = {'A':(37.5665,126.9780), 'B':(35.1796,129.0756), 'C':(35.9078,127.7669)}
    df_map = df_latest.copy()
    df_map['lat'] = df_map['group'].map(lambda x: loc_map.get(x, (np.nan,np.nan))[0])
    df_map['lon'] = df_map['group'].map(lambda x: loc_map.get(x, (np.nan,np.nan))[1])
    df_map = df_map.dropna(subset=['lat','lon'])
    if not df_map.empty:
        st.map(df_map.rename(columns={y_col:'value'}))
    else:
        st.info("지도 표시를 위한 위치 데이터가 없습니다.")

    # CSV 다운로드
    st.download_button("사용자 입력(전처리된) CSV 다운로드", data=df_user.to_csv(index=False).encode('utf-8'), file_name="user_input_preprocessed.csv", mime="text/csv")

# -----------------------
# 메인
# -----------------------
def main():
    st.title("Streamlit + GitHub Codespaces 데이터 대시보드 템플릿")
    st.markdown("한국어 UI — 공개 데이터 먼저 불러온 뒤 사용자 입력(프롬프트 제공 데이터) 기반 대시보드를 생성합니다.")
    st.sidebar.title("네비게이션")
    page = st.sidebar.radio("페이지 선택", ["공개 데이터 대시보드", "사용자 입력 대시보드", "앱 정보"])

    if page == "공개 데이터 대시보드":
        public_dashboard()
    elif page == "사용자 입력 대시보드":
        user_input_dashboard()
    else:
        st.header("앱 정보")
        st.markdown("""
        - 이 앱은 Streamlit + Codespaces 환경에서 즉시 실행 가능한 대시보드 템플릿입니다.
        - 공개 데이터(예: NOAA, NASA, World Bank)를 먼저 시도해 불러오며, 실패 시 예시 데이터로 대체하고 화면에 안내합니다.
        - 사용자 입력 데이터(프롬프트 Input 섹션으로 주어지는 CSV/이미지/설명)가 없으면 예시 데이터를 사용하여 대시보드를 자동 생성합니다.
        - 모든 라벨·툴팁·버튼은 한국어로 작성되어 있습니다.
        - 폰트는 /fonts/Pretendard-Bold.ttf 가 있으면 적용을 시도합니다.
        - 캐싱: @st.cache_data 를 사용합니다.
        - 전처리된 표는 CSV 다운로드 버튼으로 제공합니다.
        """)
        st.markdown("**주의:** 실제 공개 데이터 API는 포맷이나 URL이 바뀔 수 있습니다. 운영 환경에서는 데이터 공급자의 최신 API 문서를 참고해 URL과 파싱 로직을 업데이트하세요.")
        st.markdown("Kaggle 데이터 사용 안내(선택적): Kaggle API를 사용하려면 kaggle.json(Username/Key)을 Codespaces의 안전한 위치(~/.kaggle/kaggle.json)에 두고, `pip install kaggle` 후 `kaggle datasets download` 또는 `kaggle competitions download` 명령을 사용하세요. 자세한 내용: https://www.kaggle.com/docs/api")

if __name__ == "__main__":
    main()
