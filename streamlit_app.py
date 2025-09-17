"""
Streamlit 대시보드 (한국어 UI)
- 주제: 폭염
- 공개 데이터: NASA GISTEMP (글로벌 기온 이상값 CSV)
- 사용자 입력: 폭염 관련 글
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from datetime import datetime, timezone, timedelta
import plotly.express as px

st.set_page_config(page_title="🌡️ 폭염 대시보드", layout="wide")

# ----- Pretendard 폰트 적용 시도 -----
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

st.title("🌡️ 폭염 & 교실 영향 대시보드")

# ---------------------------
# 오늘(로컬 자정) 계산
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

        # 앞부분 설명행 스킵 후 읽기
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
# 공개 데이터 대시보드
# ---------------------------
load_result = load_gistemp()
if not load_result["ok"]:
    st.warning("공개 데이터(GISTEMP) 다운로드 실패 → 예시 데이터 사용\n오류: " + load_result.get("error", "알 수 없음"))
gistemp_df = load_result["df"]

st.subheader("공식 데이터: 기온 이상값 시계열")
col1, col2 = st.columns([3,1])
with col2:
    rolling = st.selectbox("스무딩(개월)", [1,3,6,12], index=1)
    viz_type = st.selectbox("그래프 유형", ["꺾은선","면적"], index=0)
with col1:
    df_plot = gistemp_df.copy()
    if rolling > 1:
        df_plot['value_sm'] = df_plot['value'].rolling(window=rolling, min_periods=1).mean()
        y_col = 'value_sm'
    else:
        y_col = 'value'

    # 날짜별 평균값
    df_plot = df_plot.groupby('date', as_index=False)[y_col].mean()

    if viz_type=="꺾은선":
        fig = px.line(df_plot, x='date', y=y_col, labels={'date':'날짜', y_col:'이상값(°C)'}, title="폭염 관련 기온 이상값 추이")
    else:
        fig = px.area(df_plot, x='date', y=y_col, labels={'date':'날짜', y_col:'이상값(°C)'}, title="폭염 관련 기온 이상값 추이")
    st.plotly_chart(fig, use_container_width=True)

st.download_button("공식 데이터 CSV 다운로드", gistemp_df.to_csv(index=False).encode('utf-8'),
                   file_name="gistemp_preprocessed.csv", mime="text/csv")

# ---------------------------
# 사용자 입력 대시보드
# ---------------------------
st.markdown("---")
st.header("사용자 입력: 폭염 관련 글")

USER_TEXT = """
폭염의 가장 큰 피해자는 바로 매일 학교에서 생활하는 우리 학생들이다.
창가 자리는 햇빛 때문에 찜통이 되고, 점심시간이 지나 나른해진 오후에는 교실 전체가 찜질방처럼 변한다. 
이런 환경에서는 아무리 공부를 열심히 하려고 해도 집중력이 떨어지고, 머리가 아프거나 쉽게 피로해진다. 
우리의 건강과 학습권이 폭염에 그대로 노출된 것이다. 
실제로 폭염경보가 내리는 날이면, 체육 시간은 운동장 대신 교실에서 이론 수업으로 대체되거나 아예 취소된다. 
작년 여름방학 보충수업 기간에는 너무 더워서 단축 수업을 하기도 했다. 
결국, 바다 온도 상승 → 내륙 폭염 증가 → 교실 온도 상승 → 우리의 학습권과 건강권 위협이라는 연쇄 고리가 벌어지고 있다. 
따라서 끓는 교실의 온도를 낮추는 것은 단순히 더위를 피하는 문제가 아니라, 우리의 소중한 학습권을 지키기 위한 중요한 활동이다.
"""
USER_LINK = "https://nsp.nanet.go.kr/plan/subject/detail.do?newReportChk=list&nationalPlanControlNo=PLAN0000048033"

st.write(USER_TEXT)
st.markdown(f"출처: {USER_LINK}")

# 간단 키워드 분석
st.subheader("키워드 빈도 분석")
keywords = ['폭염','교실','학생','학습권','건강','창가','점심','체육','단축','바다','내륙','온도']
counts = {k: USER_TEXT.count(k) for k in keywords}
kw_df = pd.DataFrame({"키워드":list(counts.keys()), "빈도":list(counts.values())})
kw_df = kw_df.sort_values('빈도', ascending=False).reset_index(drop=True)
fig_kw = px.bar(kw_df, x='키워드', y='빈도', labels={'빈도':'빈도수','키워드':'키워드'}, title="키워드 빈도")
st.plotly_chart(fig_kw, use_container_width=True)

# 요약 (첫 문장 + 마지막 문장)
st.subheader("간단 요약")
lines = [ln.strip() for ln in USER_TEXT.strip().split('\n') if ln.strip()]
summary = lines[0] + " ... " + lines[-1] if len(lines)>1 else lines[0]
st.info(summary)

# 사용자 입력 CSV 다운로드
st.subheader("사용자 입력 전처리 표 다운로드")
user_table = pd.DataFrame({
    '원문구분':['본문'],
    '텍스트길이':[len(USER_TEXT)],
    '주요키워드':[", ".join(kw_df[kw_df['빈도']>0]['키워드'].tolist())],
    '출처링크':[USER_LINK]
})
st.dataframe(user_table)
st.download_button("사용자 입력 CSV 다운로드", data=user_table.to_csv(index=False).encode('utf-8'),
                   file_name="user_input_preprocessed.csv", mime="text/csv")
