# streamlit_app.py
"""
Streamlit 대시보드 (한국어 UI)
- 공식 공개 데이터 대시보드: NASA GISTEMP (글로벌 기온 이상값 CSV)
  출처(코드 주석에 명시)
  - https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv
- 사용자 입력 대시보드: 사용자가 제공한 '설명 텍스트' + 링크를 기반으로 간단한 텍스트 기반 인사이트 및 시각화
요구사항 요약:
- 한국어 UI
- 오늘(로컬 자정) 이후 데이터 제거
- API/다운로드 실패 시 예시 데이터로 자동 대체 (화면에 안내)
- 전처리: 결측 처리 / 형변환 / 중복 제거 / 미래데이터 제거
- 캐시: @st.cache_data 사용
- 전처리된 표를 CSV 다운로드 버튼으로 제공
- Pretendard 폰트 시도: /fonts/Pretendard-Bold.ttf (없으면 자동 생략)
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from datetime import datetime, timezone, timedelta
import plotly.express as px

st.set_page_config(page_title="폭염 & 교실 영향 대시보드", layout="wide")

# ----- Pretendard 적용 시도 (있으면 사용, 없으면 무시) -----
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

st.title("🌡️ 폭염과 교실 — 공개 데이터 + 사용자 입력 대시보드")
st.caption("공식 공개 데이터로 먼저 분석하고, 아래에 제공된 사용자 입력 텍스트를 별도 대시보드로 보여줍니다.")

# ---------------------------
# 유틸: 오늘(로컬 자정) 계산 (Asia/Seoul)
# ---------------------------
def local_midnight_today():
    # Asia/Seoul is UTC+9
    tz_offset = 9
    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    local_now = now_utc + timedelta(hours=tz_offset)
    local_midnight = datetime(year=local_now.year, month=local_now.month, day=local_now.day)
    # convert to UTC-aware
    return local_midnight - timedelta(hours=tz_offset)

LOCAL_MIDNIGHT_UTC = local_midnight_today()

# ---------------------------
# 공개 데이터: NASA GISTEMP 가져오기
# ---------------------------
GISTEMP_CSV_URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"

@st.cache_data(ttl=60*60*6, show_spinner=False)
def load_gistemp(url=GISTEMP_CSV_URL, timeout=10):
    """
    시도 순서:
    1) 원본 CSV 요청
    2) 실패하면 예시 데이터를 반환 (그리고 플래그로 실패 알림)
    """
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        # 서버가 CSV-like 텍스트로 제공 -> 읽기
        text = resp.text
        # NASA GISTEMP 파일은 앞에 주석/빈행이 적을 수 있으므로 pandas.read_csv with skiprows heuristic
        # 하지만 file often begins with header "Land-Ocean: Global Means Year,Jan,Feb,..."
        df = pd.read_csv(io.StringIO(text), comment='#')
        # Clean: sometimes trailing columns like '***' exist; keep Year and months and J-D
        # We'll keep Year and monthly columns (Jan..Dec) and J-D (annual)
        needed_cols = [c for c in df.columns if c.strip() != ""]
        df = df.loc[:, needed_cols]
        # Rename Year column to 'Year' if needed
        if 'Year' not in df.columns and df.columns[0].lower().startswith('land'):
            # sometimes the first column header is like 'Land-Ocean: Global Means Year'
            # so try to split
            new_cols = df.columns.tolist()
            # last token likely 'Year' in header string - try to fix:
            # fallback: set first column name to 'Year'
            df = df.rename(columns={df.columns[0]: 'Year'})
        # Keep Year and J-D (annual)
        # If monthly columns present, melt to monthly timeseries
        # Identify month columns names (Jan..Dec)
        month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        available_months = [m for m in month_names if m in df.columns]
        if available_months:
            # melt to long format
            df_melt = df.melt(id_vars=['Year'], value_vars=available_months, var_name='month', value_name='anom')
            # Create date: Year-month-01
            df_melt['Year'] = df_melt['Year'].astype(int)
            month_num = {m:i+1 for i,m in enumerate(month_names)}
            df_melt['month_num'] = df_melt['month'].map(month_num)
            df_melt['date'] = pd.to_datetime(df_melt['Year']*10000 + df_melt['month_num']*100 + 1, format='%Y%m%d')
            # Convert anomalous formats like '.12' or '-.12' -> numeric
            df_melt['anom'] = pd.to_numeric(df_melt['anom'].astype(str).str.replace('*',''), errors='coerce')
            df_final = df_melt[['date','anom']].sort_values('date').reset_index(drop=True)
            df_final = df_final.rename(columns={'anom':'value'})
            df_final['group'] = 'GISTEMP월별'
        else:
            # fallback: use J-D column if exists (annual)
            if 'J-D' in df.columns:
                df2 = df[['Year','J-D']].copy()
                df2 = df2.rename(columns={'Year':'year','J-D':'value'})
                df2['date'] = pd.to_datetime(df2['year'].astype(int).astype(str) + '-01-01')
                df_final = df2[['date','value']].sort_values('date').reset_index(drop=True)
                df_final['group'] = 'GISTEMP연간'
            else:
                raise ValueError("GISTEMP 파일 포맷을 해석할 수 없음.")
        # 전처리: 결측 처리(보간 아님 — 결측 유지), 중복 제거, 형변환
        df_final = df_final.drop_duplicates(subset=['date'])
        df_final['value'] = pd.to_numeric(df_final['value'], errors='coerce')
        # 미래 데이터 제거: remove rows with date >= local midnight (UTC)
        df_final = df_final[df_final['date'] < LOCAL_MIDNIGHT_UTC]
        df_final = df_final.reset_index(drop=True)
        return {"ok": True, "df": df_final, "source": url}
    except Exception as e:
        # 예시 데이터 (간단한 월별 예시) — 사용자에게 실패를 UI에 알릴 것
        example_dates = pd.date_range(end=(LOCAL_MIDNIGHT_UTC - pd.Timedelta(days=1)), periods=60, freq='M')
        ex_df = pd.DataFrame({
            'date': example_dates,
            'value': np.linspace(0.2, 1.2, len(example_dates)) + np.random.normal(scale=0.05, size=len(example_dates)),
            'group': '예시_GISTEMP'
        })
        return {"ok": False, "df": ex_df, "error": str(e), "source": url}

# ----- 로드 및 UI 표시 -----
with st.expander("공개 데이터: NASA GISTEMP (글로벌 기온 이상값) 불러오기 · 설명", expanded=True):
    st.write("데이터 소스: NASA GISS GISTEMP (GLB.Ts+dSST). 월별 또는 연별 이상값을 사용합니다.")
    st.markdown("- 원본 URL: https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv")
    st.markdown("- 처리 규칙: 결측 처리 / 중복 제거 / '오늘(로컬 자정) 이후' 데이터 제거")
    st.markdown("- 실패 시: 예시 데이터로 자동 대체 (화면에 안내 표시)")

load_result = load_gistemp()

if not load_result["ok"]:
    st.warning("공개 데이터(GISTEMP) 다운로드에 문제가 발생했습니다. 예시 데이터로 대체합니다.\n오류: " + load_result.get("error", "알 수 없는 오류"))
gistemp_df = load_result["df"]

# 상단 요약카드
col1, col2, col3 = st.columns([2,2,2])
with col1:
    st.metric("데이터 기간 (시작)", gistemp_df['date'].min().strftime("%Y-%m-%d"))
with col2:
    st.metric("데이터 기간 (끝)", gistemp_df['date'].max().strftime("%Y-%m-%d"))
with col3:
    st.metric("샘플 수", f"{len(gistemp_df):,}")

# 주요 시각화: 시계열 (꺾은선 + 면적 선택)
st.subheader("공식 공개 데이터 대시보드 — 기온 이상값 시계열")
colA, colB = st.columns([3,1])
with colB:
    st.write("옵션")
    rolling = st.selectbox("스무딩 (이동평균 기간, 월 기준)", [1,3,6,12], index=1, help="값이 1이면 스무딩 없음")
    viz_type = st.selectbox("그래프 유형", ["꺾은선", "면적(Area)"], index=0)
    show_points = st.checkbox("데이터 점 표시", value=False)
with colA:
    df_plot = gistemp_df.copy()
    if rolling and rolling > 1:
        df_plot['value_sm'] = df_plot['value'].rolling(window=rolling, min_periods=1).mean()
        y_col = 'value_sm'
    else:
        y_col = 'value'
    title = "전지구 표면 온도 이상값 (NASA GISTEMP)"
    fig = px.line(df_plot, x='date', y=y_col, title=title, labels={'date':'날짜','value':'이상값(°C)'})
    if viz_type == "면적(Area)":
        fig = px.area(df_plot, x='date', y=y_col, title=title, labels={'date':'날짜','value':'이상값(°C)'})
    if show_points:
        fig.update_traces(mode='lines+markers')
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# 테이블과 CSV 다운로드
st.subheader("전처리된 표 (다운로드 가능)")
st.dataframe(gistemp_df.head(100))
csv_bytes = gistemp_df.to_csv(index=False).encode('utf-8')
st.download_button("전처리된 데이터 CSV 다운로드", data=csv_bytes, file_name="gistemp_preprocessed.csv", mime="text/csv")

# ---------------------------
# 사용자 입력 대시보드 (입력: 텍스트 + 링크)
# ---------------------------
st.markdown("---")
st.header("사용자 입력 대시보드 (제공된 텍스트/링크 기반)")

# 사용자 입력 (프롬프트에서 제공된 텍스트와 링크을 하드코딩하여 앱 실행 중 추가 입력 요구하지 않음)
USER_TEXT = """
폭염의 가장 큰 피해자는 바로 매일 학교에서 생활하는 우리 학생들이다.
창가 자리는 햇빛 때문에 찜통이 되고, 점심시간이 지나 나른해진 오후에는 교실 전체가 찜질방처럼 변한다. 이런 환경에서는 아무리 공부를 열심히 하려고 해도 집중력이 떨어지고, 머리가 아프거나 쉽게 피로해진다. 우리의 건강과 학습권이 폭염에 그대로 노출된 것이다.
실제로 폭염경보가 내리는 날이면, 체육 시간은 운동장 대신 교실에서 이론 수업으로 대체되거나 아예 취소된다. 작년 여름방학 보충수업 기간에는 너무 더워서 단축 수업을 하기도 했다. 결국, 바다 온도 상승 → 내륙 폭염 증가 → 교실 온도 상승 → 우리의 학습권과 건강권 위협이라는 연쇄 고리가 우리 눈앞에서 벌어지고 있는 셈이다.
따라서 끓는 교실의 온도를 낮추는 것은 단순히 더위를 피하는 문제가 아니라, 우리의 소중한 학습권을 지키기 위한 중요한 활동이다.
"""
USER_LINK = "https://nsp.nanet.go.kr/plan/subject/detail.do?newReportChk=list&nationalPlanControlNo=PLAN0000048033"

st.subheader("원문 (사용자 제공)")
st.write(USER_TEXT)
st.markdown(f"**제공 링크:** {USER_LINK}")

# 텍스트 기반 간단한 자연어 분석 (키워드 빈도)
st.subheader("텍스트 기반 인사이트 — 키워드 빈도 분석")

def simple_keyword_counts(text, keywords=None):
    # 한국어 형태소 분석을 쓰지 않고 간단 키워드 카운트 (문장에 따라 충분)
    if keywords is None:
        keywords = ['폭염','교실','학생','학습권','건강','창가','점심','체육','단축','바다','내륙','온도']
    lowered = text.replace('\n',' ').lower()
    counts = {k: lowered.count(k) for k in keywords}
    dfk = pd.DataFrame({"키워드":list(counts.keys()), "빈도":list(counts.values())})
    dfk = dfk.sort_values('빈도', ascending=False).reset_index(drop=True)
    return dfk

kw_df = simple_keyword_counts(USER_TEXT)
fig_kw = px.bar(kw_df, x='키워드', y='빈도', title="키워드 빈도 (간단 카운트)", labels={'빈도':'빈도수','키워드':'키워드'})
st.plotly_chart(fig_kw, use_container_width=True)

# 간단 텍스트 요약(핵심 문장 추출) - rule-based: 첫 문장 + 결론문
st.subheader("간단 요약 (자동 생성)")
lines = [ln.strip() for ln in USER_TEXT.strip().split('\n') if ln.strip()]
summary = ""
if lines:
    summary = lines[0]
    if len(lines) > 1:
        summary += " ... " + lines[-1]
st.info(summary)

# 사용자 입력 기반 추천 지표 (예시)
st.subheader("권장 지표 (학교 관점)")
st.write("- 교실 실내온도(시계열): 수업 시간대별 측정 권장")
st.write("- 학생 체감(건강) 지표: 두통/졸림/집중저하 발생률 조사")
st.write("- 대체 수업/단축수업 발생 빈도: 폭염일수와의 상관분석")

# 사용자 입력 전처리 표 다운로드 (텍스트를 표로 변환한 예시)
st.subheader("사용자 입력 전처리 표 (다운로드)")
user_table = pd.DataFrame({
    '원문구분':['본문'],
    '텍스트길이': [len(USER_TEXT)],
    '주요키워드': [", ".join(kw_df[kw_df['빈도']>0]['키워드'].tolist())],
    '출처링크':[USER_LINK]
})
st.dataframe(user_table)
st.download_button("사용자 입력 전처리 CSV 다운로드", data=user_table.to_csv(index=False).encode('utf-8'), file_name='user_input_preprocessed.csv', mime='text/csv')

# ---------------------------
# 마무리 / 참고문헌
# ---------------------------
st.markdown("---")
st.subheader("참고 및 데이터 출처")
st.markdown("""
- NASA GISS GISTEMP (GLB.Ts+dSST.csv) — Global Land-Ocean Temperature Index (monthly & annual).  
  URL: https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv  
  (코드에서 자동 다운로드 시도 — 실패 시 예시 데이터로 대체됩니다.)
- NOAA Climate at a Glance / OISST 등 — 추가 분석시 활용 가능. (예: https://www.ncei.noaa.gov)
""")

st.caption("앱 구현 규칙: date / value / group 표준화, 결측 처리, 미래 데이터(오늘 이후) 제거, @st.cache_data 사용, CSV 다운로드 제공.")
