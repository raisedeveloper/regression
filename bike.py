import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from ydata_profiling import ProfileReport

# 데이터 불러오기
data_path = 'dataset/bike.csv'
df = pd.read_csv(data_path)

# datetime 변환 및 날짜 관련 특징 생성
df['datetime'] = pd.to_datetime(df['datetime'])
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour
df['weekday'] = df['datetime'].dt.weekday
df.drop(columns=['datetime'], inplace=True)

# casual과 registered 컬럼이 있는지 확인하고 처리
if 'casual' in df.columns and 'registered' in df.columns:
    X = df.drop(columns=['count', 'casual', 'registered'])
else:
    X = df.drop(columns=['count'])

y = df['count']

# 모델 학습에 사용할 특성명 저장
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

plt.figure(figsize=(10, 6))
plt.bar(model_names, r2_scores, color='skyblue')
plt.xlabel("모델")
plt.ylabel("R2 Score")
plt.title("모델별 R2 Score 비교")
plt.grid(axis='y')
plt.show()

# Streamlit 앱 설정
st.title("🚲 공공자전거 대여량 예측")
st.write(f'모델 성능 - RMSE: {rmse:.2f}, R-squared: {r2:.4f}')

# 사용자 입력 받기
season = st.selectbox('계절 (1:봄, 2:여름, 3:가을, 4:겨울)', [1, 2, 3, 4])
holiday = st.selectbox('공휴일 여부 (0:아님, 1:공휴일)', [0, 1])
workingday = st.selectbox('근무일 여부 (0:아님, 1:근무일)', [0, 1])
weather = st.selectbox('날씨 (1:맑음, 2:흐림, 3:비/눈)', [1, 2, 3])
temp = st.slider('온도 (0~1)', 0.0, 1.0, 0.5)
atemp = st.slider('체감 온도 (0~1)', 0.0, 1.0, 0.48)
humidity = st.slider('습도 (0~100)', 0, 100, 50)
windspeed = st.slider('풍속 (0~1)', 0.0, 1.0, 0.2)
year = st.selectbox('년도 (2011~2012)', [2011, 2012])
month = st.slider('월 (1~12)', 1, 12, 6)
day = st.slider('일 (1~31)', 1, 31, 15)
hour = st.slider('시간 (0~23)', 0, 23, 10)
weekday = st.slider('요일 (0:월~6:일)', 0, 6, 2)

# 예측 버튼 추가
if st.button("예측하기"):
    # 입력 특성 준비
    features = {
        'season': season, 'holiday': holiday, 'workingday': workingday, 'weather': weather,
        'temp': temp, 'atemp': atemp, 'humidity': humidity, 'windspeed': windspeed,
        'year': year, 'month': month, 'day': day, 'hour': hour, 'weekday': weekday
    }
    
    # 훈련에 사용된 특성과 일치하는 입력 데이터프레임 생성
    input_features = {}
    for feature in feature_names:
        if feature in features:
            input_features[feature] = features[feature]
        else:
            st.error(f"필요한 특성 '{feature}'가 입력에 없습니다.")
            break
    
    if len(input_features) == len(feature_names):
        features_df = pd.DataFrame([input_features])
        
        # 간단히 예측만 수행
        prediction = rf_model.predict(features_df)[0]
        st.success(f'예측된 대여량: {prediction:.2f}대')
        
import os
import pandas as pd
from ydata_profiling import ProfileReport

# 데이터 불러오기
file_path = 'dataset/bike.csv'
df = pd.read_csv(file_path)

# 데이터 정보 출력
print(df.info())

# y-frofiling 이용 프로파일링 리포트 생성
profile = ProfileReport(
    df,
    title="bike EDA 보고서",
    explorative=True,
    html={
        'style': {
            'theme': 'united'  # 허용된 theme 중 하나로 변경
        }
    }
)

# 보고서를 HTML 파일로 저장
output_file = 'report/bike_report.html'
profile.to_file(output_file)
print(f"프로파일링 보고서가 생성되었습니다: {output_file}")
