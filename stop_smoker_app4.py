import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import altair as alt

if 'predicted' not in st.session_state:
    st.session_state['predicted'] = False

if 'prediction_proba' not in st.session_state:
    st.session_state['prediction_proba'] = None

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="금연 성공 예측기", layout="wide")
st.title("🚭금연 성공 가능성 예측기 🚭")

# CSV 데이터 로드 및 전처리 함수
@st.cache_data
def load_and_preprocess_data():
    try:
        # MySQL 연결 대신 CSV 파일 직접 로드
        df = pd.read_csv('smoke_ulsan.csv')

        df.columns = ['provider_type', 'region', 'service_type', 'provider', 'birth_year_group', 'gender', 'reg_year', 'reg_month',
                      'reg_type', 'quit_year', 'quit_month', 'counseling_count', 'status', 'completion_year', 'completion_month',
                      '4w_success', '4w_method', 'co_4w', 'cot_4w', '6w_success', '6w_method', 'co_6w', 'cot_6w', '12w_success',
                      '12w_method', 'co_12w', 'cot_12w', '6M_success', '6M_method', 'co_6M', 'cot_6M']

        df['success_6M'] = df['6M_success'].apply(lambda x: 1 if x == 'Y' else 0)
        df['counseling_count'] = pd.to_numeric(df['counseling_count'], errors='coerce')
        df['quit_period'] = (df['completion_year'] - df['quit_year']) * 12 + (df['completion_month'] - df['quit_month'])

        df.dropna(subset=['provider', 'birth_year_group', 'gender', 'counseling_count', 'success_6M', 'quit_period'], inplace=True)
        
        df_model = df[['provider', 'birth_year_group', 'gender', 'counseling_count', 'quit_period', 'success_6M']].copy()
        df_model = pd.get_dummies(df_model, columns=['provider', 'birth_year_group', 'gender'], drop_first=True)
        
        X = df_model.drop('success_6M', axis=1)
        y = df_model['success_6M']
        
        return df, X, y
    except Exception as e:
        st.error(f"❌ 데이터 로드 또는 전처리 실패: {e}")
        st.stop()
        return None, None, None

df, X, y = load_and_preprocess_data()

# 데이터 로딩 실패 시 앱 중단
if df is None:
    st.stop()
    
# 로지스틱 회귀 모델 학습
X_sm = sm.add_constant(X).astype(float)
y_sm = y.astype(float)
logit_model = sm.Logit(y_sm, X_sm).fit(disp=False)

# 탭 구성
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 데이터 미리보기", "📈 시각화", "🔮 예측", "📝 데이터 입력", "💡 금연 정보", "🧠 AI 추천 전략"
])

with tab1:
    st.markdown("### 원본 데이터 미리보기")
    providers = sorted(df['provider'].unique())
    selected_provider = st.selectbox("📍 보건소를 선택하세요:", providers)
    filtered_df = df[df['provider'] == selected_provider]
    row_count = st.slider("표시할 행 수:", 1, len(filtered_df), min(10, len(filtered_df)), key='row_count_slider_1')
    st.dataframe(filtered_df.head(row_count))

    st.markdown("### 전처리된 데이터 미리보기")
    st.dataframe(X.head())

with tab2:
    st.markdown("### 성별에 따른 6개월 금연 성공률")
    fig1, ax1 = plt.subplots(figsize=(5, 3))
    sns.barplot(x='gender', y='success_6M', data=df, ax=ax1)
    st.pyplot(fig1)

    st.markdown("### 상담 횟수 분포")
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    sns.histplot(df['counseling_count'], bins=20, kde=True, ax=ax2)
    st.pyplot(fig2)

    st.markdown("### 변수 간 상관관계 히트맵 (인터랙티브)")
    st.info("💡 마우스를 올리면 상관관계 값을 확인할 수 있습니다.")
    
    # Altair 히트맵을 위한 데이터 전처리
    corr_df = df_model.corr().reset_index().rename(columns={'index': 'variable1'})
    corr_df = corr_df.melt('variable1', var_name='variable2', value_name='correlation')

    # Altair 차트 생성
    chart = alt.Chart(corr_df).mark_rect().encode(
        x=alt.X('variable1', title='변수'),
        y=alt.Y('variable2', title='변수'),
        color=alt.Color('correlation', legend=alt.Legend(title="상관관계")),
        tooltip=[
            alt.Tooltip('variable1', title='변수 1'),
            alt.Tooltip('variable2', title='변수 2'),
            alt.Tooltip('correlation', title='상관관계 값', format='.2f')
        ]
    ).properties(
        title='변수 간 상관관계 히트맵'
    )
    st.altair_chart(chart, use_container_width=True)


with tab3:
    st.markdown("### 로지스틱 회귀 분석 결과")
    st.text(logit_model.summary().as_text())

    st.write("---")
    st.markdown("### 🔮 금연 성공 가능성 예측")
    st.markdown("아래 변수들을 조절하여 금연 성공 가능성을 예측해 보세요.")

    # 예측에 사용할 입력 위젯
    gender_map = {'남성': '남', '여성': '여'}
    providers = sorted(df['provider'].unique())
    birth_year_groups = sorted(df['birth_year_group'].unique())

    selected_provider = st.selectbox('📍 보건소를 선택하세요:', providers, key='predict_provider')
    gender_selection = st.selectbox('성별을 선택하세요:', ['남성', '여성'], key='predict_gender')
    birth_year_selection = st.selectbox('출생년도 그룹을 선택하세요:', birth_year_groups, key='predict_birth_year')
    min_counseling = int(df['counseling_count'].min())
    max_counseling = int(df['counseling_count'].max())
    counseling_count = st.slider('상담 횟수를 조절하세요:', min_value=min_counseling, max_value=max_counseling, value=10, key='predict_counseling')
    min_period = int(df['quit_period'].min())
    max_period = int(df['quit_period'].max())
    quit_period = st.slider('금연 결심 기간 (개월):', min_value=min_period, max_value=max_period, value=6, key='predict_quit_period')

    if st.button('예측하기', key='predict_button'):
        # 1. 예측 로직 실행
        input_data = pd.DataFrame(0, index=[0], columns=X.columns)
        input_data['counseling_count'] = counseling_count
        input_data['quit_period'] = quit_period
        
        # 더미 변수 설정
        if 'gender_여' in input_data.columns:
            input_data['gender_여'] = 1 if gender_selection == '여성' else 0
        if f'provider_{selected_provider}' in input_data.columns:
            input_data[f'provider_{selected_provider}'] = 1
        if f'birth_year_group_{birth_year_selection}' in input_data.columns:
            input_data[f'birth_year_group_{birth_year_selection}'] = 1

        input_data_with_const = sm.add_constant(input_data, has_constant='add')
        prediction_proba = logit_model.predict(input_data_with_const)[0]
        
        # 예측 결과에 따라 status 값 결정
        prediction_status = '성공예측' if prediction_proba >= 0.5 else '실패예측'
        
        # 2. 예측 결과 저장 및 시각화
        st.session_state['prediction_proba'] = prediction_proba
        st.session_state['predicted'] = True
        st.session_state['success_rate'] = prediction_proba
        st.session_state['counseling_count'] = counseling_count
        st.session_state['quit_period'] = quit_period

        st.markdown("#### 예측 결과")
        st.info(f"선택한 조건에서의 금연 성공 확률: **{prediction_proba*100:.2f}%**")
        st.progress(prediction_proba)

        # 예측 확률 시각화 (로지스틱 회귀 곡선)
        st.markdown("#### 상담 횟수 vs 금연 성공 가능성 그래프")
        
        base_input = pd.DataFrame(0, index=[0], columns=X.columns)
        base_input['quit_period'] = quit_period
        
        if 'gender_여' in base_input.columns:
            base_input['gender_여'] = 1 if gender_selection == '여성' else 0
        if f'provider_{selected_provider}' in base_input.columns:
            base_input[f'provider_{selected_provider}'] = 1
        if f'birth_year_group_{birth_year_selection}' in base_input.columns:
            base_input[f'birth_year_group_{birth_year_selection}'] = 1

        counts_range = np.arange(min_counseling, max_counseling + 1)
        selected_periods = [3, 6, 9, 12]
        fig, ax = plt.subplots(figsize=(6, 4))

        for period in selected_periods:
            probabilities = []
            for count in counts_range:
                temp_data = base_input.copy()
                temp_data['counseling_count'] = count
                temp_data['quit_period'] = period
                temp_data_with_const = sm.add_constant(temp_data, has_constant='add')
                prob = logit_model.predict(temp_data_with_const)[0]
                probabilities.append(prob)
            
            ax.plot(counts_range, probabilities, label=f'금연 결심 {period}개월')

        sns.scatterplot(x=df['counseling_count'], y=df['success_6M'], alpha=0.3, label='실제 데이터', ax=ax)
        ax.plot(counseling_count, prediction_proba, 'go', markersize=10, label='현재 예측값')

        ax.set_title('상담 횟수에 따른 금연 성공 확률 (기간별)')
        ax.set_xlabel('상담 횟수')
        ax.set_ylabel('금연 성공 확률')
        ax.legend()
        ax.set_ylim(-0.1, 1.1)

        st.pyplot(fig)


        # 변수 중요도 대시보드 및 동기 부여 대시보드 추가
        st.write("---")
        st.markdown("### 📈 변수 중요도 분석")
        st.info("모델에서 금연 성공에 가장 큰 영향을 미치는 변수들입니다.")
        
        # 변수 중요도 계산 (회귀 계수의 절대값)
        importance = logit_model.params.drop('const').abs().sort_values(ascending=True)
        
        fig_imp, ax_imp = plt.subplots(figsize=(10, len(importance) * 0.5))
        importance.plot(kind='barh', ax=ax_imp, color='skyblue')
        ax_imp.set_title('변수 중요도 (회귀 계수 절댓값)')
        ax_imp.set_xlabel('중요도 (계수 절댓값)')
        ax_imp.set_ylabel('변수')
        st.pyplot(fig_imp)

        st.write("---")
        st.markdown("### 💪 동기 부여 대시보드")
        
        col_input1, col_input2 = st.columns(2)
        avg_cigarettes = col_input1.number_input("하루 평균 흡연량 (개비):", min_value=1, value=20, key='avg_cigarettes')
        pack_price = col_input2.number_input("담배 한 갑 가격 (원):", min_value=1000, value=4500, key='pack_price')
        
        if st.button('동기 부여 계산하기'):
            if 'quit_period' in st.session_state:
                days_quit = st.session_state['quit_period'] * 30
                money_saved = (days_quit * avg_cigarettes / 20) * pack_price
                time_saved = (days_quit * avg_cigarettes * 5) # 1개비 당 5분 가정

                st.markdown(f"#### 당신의 금연 기간 동안...")
                st.metric(label="금연 일수", value=f"{days_quit}일 🚭")
                st.metric(label="절약한 금액", value=f"💰 {money_saved:,.0f}원")
                st.metric(label="절약한 시간", value=f"⏰ {time_saved/60:.2f}시간")


with tab4:
    st.markdown("### 📝 금연 시도 정보 저장")
    st.markdown("새로운 금연 시도자의 정보를 입력하고 데이터베이스에 저장합니다.")
    st.warning("⚠️ 테스트 모드이므로 데이터베이스에 저장되지 않습니다.")

    with st.form("new_smoker_form"):
        providers = sorted(df['provider'].unique())
        birth_year_groups = sorted(df['birth_year_group'].unique())

        new_provider = st.selectbox('보건소:', providers, key='new_provider')
        new_birth_year = st.selectbox('출생년도 그룹:', birth_year_groups, key='new_birth_year')
        new_gender = st.selectbox('성별:', ['남', '여'], key='new_gender')
        new_counseling = st.number_input('상담 횟수:', min_value=1, value=1, key='new_counseling')
        
        submitted = st.form_submit_button("데이터 저장하기")
        if submitted:
            st.error("❌ 현재 테스트 모드이므로 데이터를 저장할 수 없습니다.")

with tab5:
    st.markdown("## 💡 금연을 위한 정보와 전략")
    st.markdown("### 🔥 왜 금연해야 할까?")
    st.write("""
    - 담배에는 **7,000종 이상의 유해 화학물질**이 포함되어 있으며, 그 중 **70종 이상이 발암물질**입니다.
    - 대표적인 유해 성분:
        - **니코틴**: 중독 유발
        - **타르**: 치아 변색, 폐 손상
        - **일산화탄소**: 산소 부족, 만성 피로
        - 기타: 아세톤, 나프타린, 청산가스, 벤조피렌 등
    > 흡연은 본인의 건강뿐 아니라 **가족과 주변 사람의 건강에도 직접적인 위협**이 됩니다.
    """)

    st.markdown("### 💰 흡연의 경제적 손실")
    st.write("""
    - 하루 1갑 기준 월 약 **15~20만 원 지출**
    - 흡연자는 비흡연자에 비해 **병원비와 건강 유지 비용이 더 많이 듭니다**
    """)

    st.markdown("### 🚧 금연의 어려움과 방해 요소")
    st.table(pd.DataFrame({
        "요소": ["중독성", "스트레스", "체중 증가", "사회적 유혹"],
        "설명": [
            "니코틴은 마약 수준의 중독성을 가짐",
            "금단 증상과 함께 재흡연 유도",
            "특히 여성 흡연자에게 심리적 저항 요인",
            "주변 흡연자, 술자리 등 환경적 요인"
        ]
    }))

    st.markdown("### 🧪 금연 보조제 및 치료법")
    st.markdown("#### 1. 니코틴 대체 요법")
    st.table(pd.DataFrame({
        "제품": ["금연 껌", "니코틴 패치"],
        "특징": ["입 안 점막 통해 니코틴 공급", "피부 통해 니코틴 공급"],
        "사용법": ["30분 씹기, 하루 15회 이하", "하루 1매, 8~12주 사용"]
    }))

    st.markdown("#### 2. 금연 처방약")
    st.write("""
    - 금연 클리닉에서 처방
    - 니코틴 없이 흡연 욕구 억제
    - 복용 시작 1~2주 전부터 단계적 투여
    """)

    st.markdown("### 📈 금연 후 신체 변화")
    st.table(pd.DataFrame({
        "시간 경과": ["20분 후", "12시간 후", "2주~3개월", "1년 후", "5년 후", "10년 후"],
        "변화": [
            "혈압과 심박수 정상화",
            "혈중 일산화탄소 농도 정상",
            "폐기능 회복, 혈액순환 개선",
            "심장병 위험 절반으로 감소",
            "뇌졸중 위험 비흡연자 수준",
            "폐암 위험 절반으로 감소"
        ]
    }))

    st.markdown("### 🏥 전국 금연 클리닉 정보")
    st.write("""
    - 보건소 금연클리닉은 무료로 상담과 니코틴 대체요법을 제공합니다.
    - 가까운 보건소를 찾아 방문해보세요.
    """)
    st.markdown("- [📍 전국 보건소 금연클리닉 찾기](https://www.nosmokeguide.go.kr/lay2/program/S1T53C107/nosmoke/centermap/bogun_list.do)")
    st.write("- 금연상담전화: **1544-9030** (무료 상담 및 금연 계획 수립 지원)")

with tab6:
    st.markdown("## 🧠 AI 기반 금연 전략 추천")

    if 'success_rate' in st.session_state and 'counseling_count' in st.session_state:
        rate = st.session_state['success_rate']
        count = st.session_state['counseling_count']
        
        st.markdown(f"### 예측된 금연 성공 확률: `{rate*100:.2f}%`")
        st.markdown("#### 당신의 성공률에 기반한 맞춤 전략을 제안합니다:")
        
        if rate < 0.4:
            st.error("성공률이 낮습니다. 다음 전략을 강력히 추천합니다:")
            st.write("- **니코틴 대체요법**: 패치, 껌 등 사용")
            st.write("- **금연 앱 활용**: '금연두드림', '금연길라잡이' 등으로 기록과 동기부여")
            st.write("- **금연 클리닉 방문**: 전문가 상담 및 처방약 지원")
            st.write("- **흡연 유발 환경 제거**: 술자리, 스트레스 상황 피하기")
            st.write("- **AI 금연 서비스 활용**: [금연길라잡이 AI검색요약 서비스](https://www.nosmokeguide.go.kr)에서 맞춤 정보 받기")
        elif rate < 0.7:
            st.warning("중간 수준입니다. 다음 전략을 추천합니다:")
            st.write("- **상담 횟수 늘리기**: 주 1회 이상 상담 유지")
            st.write("- **스트레스 관리**: 명상, 운동, 심호흡 등으로 대체 행동 훈련")
            st.write("- **보상 시스템 만들기**: 금연 일수에 따라 스스로에게 선물하기")
            st.write("- **금연 동료 만들기**: 함께 금연하는 친구 또는 커뮤니티 참여")
        else:
            st.success("성공 가능성이 높습니다! 유지 전략을 추천합니다:")
            st.write("- **금연 일지 작성**: 금연 이유, 변화, 감정 기록하기")
            st.write("- **가족·친구에게 알리기**: 지지 환경 조성")
            st.write("- **건강검진 예약**: 금연 후 신체 변화 확인하며 동기 강화")
            st.write("- **재흡연 방지 플랜 수립**: 유혹 상황에서 대처할 행동 미리 정하기")

        st.markdown("### 📡 최신 AI 금연 서비스 소개")
        st.write("""
        - 보건복지부와 한국건강증진개발원이 운영하는 **금연길라잡이 AI검색요약 서비스**는
          사용자의 질문 의도를 이해하고, 흡연예방 및 금연 관련 정보를 요약 제공하는 초거대 AI 기반 서비스입니다.
        - 최신 기술인 **검색증강생성(RAG)**을 활용해 신뢰도 높은 정보를 제공합니다.
        """)
        st.markdown("- [🔗 금연길라잡이 AI검색요약 서비스 바로가기](https://www.nosmokeguide.go.kr)")

        st.markdown("### 🎯 AI 추천 전략 요약")
        st.table(pd.DataFrame({
            "전략 유형": ["행동 전략", "심리 전략", "기술 활용", "환경 조성"],
            "추천 내용": [
                "니코틴 대체요법, 상담 횟수 증가",
                "스트레스 관리, 보상 시스템",
                "금연 앱, AI 검색요약 서비스",
                "흡연 유발 환경 제거, 지지자 확보"
            ]
        }))
    else:
        st.info("먼저 예측 탭에서 '예측하기' 버튼을 눌러주세요.")
