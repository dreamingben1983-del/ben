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
        # 인코딩을 'cp949'로 지정하여 파일을 로드
        df = pd.read_csv('stop_smoker.csv', encoding='cp949')

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
        st.session_state['counseling_count'] = counseling

