import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import mysql.connector
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

# MySQL 데이터 로드 함수
@st.cache_data
def load_data():
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='mysqlbig',
            database='stop_smoking'
        )
        query = "SELECT * FROM stop_smoker"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"❌ MySQL 연결 또는 데이터 로드 실패: {e}")
        st.stop()
        return None

# 기존 데이터 저장 함수
def save_data(data):
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='mysqlbig',
            database='stop_smoking'
        )
        cursor = conn.cursor()
        
        query = """INSERT INTO stop_smoker (
                     provider, birth_year_group, gender, counseling_count, status, `6M`
                     ) VALUES (%s, %s, %s, %s, %s, %s)"""
        
        cursor.execute(query, data)
        conn.commit()
        conn.close()
        st.success("✔ 데이터가 성공적으로 저장되었습니다!")
    except Exception as e:
        st.error(f"❌ 데이터 저장 실패: {e}")

# 사용자 입력 데이터 저장 함수 (users 테이블)
def save_user_data(data):
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='mysqlbig',
            database='stop_smoking'
        )
        cursor = conn.cursor()
        
        # 'prediction'과 'status' 열을 추가하여 쿼리 수정
        query = """INSERT INTO users (
                     gender, birth_year_group, counseling_count, quit_period, prediction, status
                     ) VALUES (%s, %s, %s, %s, %s, %s)"""
        
        cursor.execute(query, data)
        conn.commit()
        conn.close()
        st.success("✔ 사용자 입력 데이터가 'users' 테이블에 성공적으로 저장되었습니다!")
    except Exception as e:
        st.error(f"❌ 사용자 데이터 저장 실패: {e}")

# 데이터 로드
df = load_data()

# 전처리
df.columns = ['provider', 'birth_year_group', 'gender', 'quit_year', 'quit_month',
              'counseling_count', 'status', 'completion_year', 'completion_month',
              '4w', '6w', '12w', '6M']
df['success_6M'] = df['6M'].apply(lambda x: 1 if x == 'Y' else 0)
df['counseling_count'] = pd.to_numeric(df['counseling_count'], errors='coerce')
# 'quit_period'는 동기 부여 대시보드를 위해 유지
df['quit_period'] = (df['completion_year'] - df['quit_year']) * 12 + (df['completion_month'] - df['quit_month'])

# 모델링에 필요한 컬럼에서 결측치 제거 ('quit_period'는 모델에서 제외)
df.dropna(subset=['provider', 'birth_year_group', 'gender', 'counseling_count', 'success_6M', '4w', '6w', '12w'], inplace=True)
# '4w', '6w', '12w' 변수를 모델링에 추가
df_model = df[['provider', 'birth_year_group', 'gender', 'counseling_count', '4w', '6w', '12w', 'success_6M']].copy()
df_model = pd.get_dummies(df_model, columns=['provider', 'birth_year_group', 'gender', '4w', '6w', '12w'], drop_first=True)
X = df_model.drop('success_6M', axis=1)
y = df_model['success_6M']

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
    st.dataframe(df_model.head())

with tab2:
    st.markdown("### 성별에 따른 6개월 금연 성공률")
    fig1, ax1 = plt.subplots(figsize=(5, 3))
    sns.barplot(x='gender', y='success_6M', data=df, ax=ax1)
    st.pyplot(fig1)

    st.markdown("### 상담 횟수 분포")
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    sns.histplot(df['counseling_count'], bins=20, kde=True, ax=ax2)
    st.pyplot(fig2)

    st.markdown("### 변수 간 상관관계 히트맵")
    corr = df_model.corr()
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax3)
    ax3.set_title('변수 간 상관관계 히트맵')
    st.pyplot(fig3)

with tab3:
    st.markdown("### 로지스틱 회귀 분석 결과")
    X_sm = sm.add_constant(X).astype(float)
    y_sm = y.astype(float)
    logit_model = sm.Logit(y_sm, X_sm).fit(disp=False)
    st.text(logit_model.summary().as_text())

    st.write("---")
    st.markdown("### 🔮 금연 성공 가능성 예측")
    st.markdown("아래 변수들을 조절하여 금연 성공 가능성을 예측해 보세요.")

    gender_map = {'남성': '남', '여성': '여'}
    providers = sorted(df['provider'].unique())
    birth_year_groups = sorted(df['birth_year_group'].unique())
    eval_options = ['미실시', 'Y']

    selected_provider = st.selectbox('📍 보건소를 선택하세요:', providers, key='predict_provider')
    gender_selection = st.selectbox('성별을 선택하세요:', ['남성', '여성'], key='predict_gender')
    birth_year_selection = st.selectbox('출생년도 그룹을 선택하세요:', birth_year_groups, key='predict_birth_year')
    min_counseling = int(df['counseling_count'].min())
    max_counseling = int(df['counseling_count'].max())
    counseling_count = st.slider('상담 횟수를 조절하세요:', min_value=min_counseling, max_value=max_counseling, value=10, key='predict_counseling')
    
    st.markdown("---")
    st.markdown("### 📝 기간별 평가 성공 여부 (모델 변수)")
    st.info("실제 평가가 이루어졌다면, 해당 결과를 선택해 주세요.")
    col_eval1, col_eval2, col_eval3 = st.columns(3)
    eval_4w = col_eval1.selectbox('4주차 평가:', eval_options, key='eval_4w')
    eval_6w = col_eval2.selectbox('6주차 평가:', eval_options, key='eval_6w')
    eval_12w = col_eval3.selectbox('12주차 평가:', eval_options, key='eval_12w')

    st.markdown("---")
    st.markdown("### 💪 동기 부여 대시보드 (계산용 변수)")
    min_period = int(df['quit_period'].min())
    max_period = int(df['quit_period'].max())
    quit_period_for_dash = st.slider('금연 결심 기간 (개월):', min_value=min_period, max_value=max_period, value=6, key='quit_period_dash')
    
    col_input1, col_input2 = st.columns(2)
    avg_cigarettes = col_input1.number_input("하루 평균 흡연량 (개비):", min_value=1, value=20, key='avg_cigarettes')
    pack_price = col_input2.number_input("담배 한 갑 가격 (원):", min_value=1000, value=4500, key='pack_price')


    if st.button('예측하기', key='predict_button'):
        # 1. 예측 로직 실행
        input_data = pd.DataFrame(0, index=[0], columns=X.columns)
        input_data['counseling_count'] = counseling_count
        
        # 더미 변수 설정
        if 'gender_여' in input_data.columns:
            input_data['gender_여'] = 1 if gender_selection == '여성' else 0
        if f'provider_{selected_provider}' in input_data.columns:
            input_data[f'provider_{selected_provider}'] = 1
        if f'birth_year_group_{birth_year_selection}' in input_data.columns:
            input_data[f'birth_year_group_{birth_year_selection}'] = 1
            
        # 평가 변수 더미화
        if f'4w_{eval_4w}' in input_data.columns:
            input_data[f'4w_{eval_4w}'] = 1
        if f'6w_{eval_6w}' in input_data.columns:
            input_data[f'6w_{eval_6w}'] = 1
        if f'12w_{eval_12w}' in input_data.columns:
            input_data[f'12w_{eval_12w}'] = 1

        input_data_with_const = sm.add_constant(input_data, has_constant='add')
        prediction_proba = logit_model.predict(input_data_with_const)[0]
        
        # 예측 결과에 따라 status 값 결정
        prediction_status = '성공예측' if prediction_proba >= 0.5 else '실패예측'

        # 사용자 입력 데이터를 users 테이블에 저장
        # quit_period_for_dash를 save_user_data에 전달
        save_user_data((gender_map[gender_selection], birth_year_selection, counseling_count, quit_period_for_dash, prediction_proba, prediction_status))

        # 예측 결과 저장 및 시각화
        st.session_state['prediction_proba'] = prediction_proba
        st.session_state['predicted'] = True
        st.session_state['success_rate'] = prediction_proba
        st.session_state['counseling_count'] = counseling_count
        st.session_state['quit_period'] = quit_period_for_dash # 동기 부여 대시보드를 위해 세션 상태에 저장

        st.markdown("#### 예측 결과")
        st.info(f"선택한 조건에서의 금연 성공 확률: **{prediction_proba*100:.2f}%**")
        st.progress(prediction_proba)

        # 예측 확률 시각화
        st.markdown("#### 상담 횟수에 따른 금연 성공 가능성 (평가별)")
        
        # 그래프를 그리기 위한 시나리오 설정
        scenarios = {
            '4주/12주 모두 실패': {'4w_Y': 0, '12w_Y': 0},
            '4주만 성공': {'4w_Y': 1, '12w_Y': 0},
            '12주까지 성공': {'4w_Y': 1, '12w_Y': 1}
        }
        
        fig, ax = plt.subplots(figsize=(4, 2))
        counts_range = np.arange(min_counseling, max_counseling + 1)
        
        for name, scenario in scenarios.items():
            probabilities = []
            for count in counts_range:
                temp_data = pd.DataFrame(0, index=[0], columns=X.columns)
                temp_data['counseling_count'] = count
                temp_data['gender_여'] = 1 if gender_selection == '여성' else 0
                if f'provider_{selected_provider}' in temp_data.columns:
                    temp_data[f'provider_{selected_provider}'] = 1
                if f'birth_year_group_{birth_year_selection}' in temp_data.columns:
                    temp_data[f'birth_year_group_{birth_year_selection}'] = 1
                
                if '4w_Y' in temp_data.columns:
                    temp_data['4w_Y'] = scenario['4w_Y']
                if '12w_Y' in temp_data.columns:
                    temp_data['12w_Y'] = scenario['12w_Y']

                temp_data_with_const = sm.add_constant(temp_data, has_constant='add')
                prob = logit_model.predict(temp_data_with_const)[0]
                probabilities.append(prob)
            
            ax.plot(counts_range, probabilities, label=name)

        ax.plot(counseling_count, prediction_proba, 'go', markersize=10, label='현재 예측값')

        ax.set_title('상담 횟수에 따른 금연 성공 확률 (평가별)')
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
        
        if 'quit_period' in st.session_state and st.session_state['quit_period'] is not None:
            days_quit = st.session_state['quit_period'] * 30
            money_saved = (days_quit * avg_cigarettes / 20) * pack_price
            time_saved = (days_quit * avg_cigarettes * 5) # 1개비 당 5분 가정

            st.markdown(f"#### 당신의 금연 기간 동안...")
            st.metric(label="금연 일수", value=f"{days_quit}일 🚭")
            st.metric(label="절약한 금액", value=f"💰 {money_saved:,.0f}원")
            st.metric(label="절약한 시간", value=f"⏰ {time_saved/60:.2f}시간")
        else:
            st.info("금연 결심 기간을 입력하고 예측하기 버튼을 누르면 계산됩니다.")


with tab4:
    st.markdown("### 📝 금연 시도 정보 저장")
    st.markdown("새로운 금연 시도자의 정보를 입력하고 데이터베이스에 저장합니다.")

    with st.form("new_smoker_form"):
        providers = sorted(df['provider'].unique())
        birth_year_groups = sorted(df['birth_year_group'].unique())

        new_provider = st.selectbox('보건소:', providers, key='new_provider')
        new_birth_year = st.selectbox('출생년도 그룹:', birth_year_groups, key='new_birth_year')
        new_gender = st.selectbox('성별:', ['남', '여'], key='new_gender')
        new_counseling = st.number_input('상담 횟수:', min_value=1, value=1, key='new_counseling')
        
        submitted = st.form_submit_button("데이터 저장하기")
        if submitted:
            data_to_save = (new_provider, new_birth_year, new_gender, new_counseling, '중간종결', '미실시')
            save_data(data_to_save)

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
        st.info("먼저 예측 탭에서 '예측하기' 버튼을 눌러주세요.")   #streamlit run d:/Python/Ben/stop_smoker_app2.py