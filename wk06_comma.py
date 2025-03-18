import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# 1. 데이터 로드 및 캐싱 적용
@st.cache_data
def load_data():
    df = pd.read_csv('dataset/HR_comma_sep.csv', encoding='utf-8')
    df.rename(columns={'Departments ': 'Departments'}, inplace=True)
    df = pd.get_dummies(df, columns=['Departments', 'salary'], drop_first=True)
    return df

df = load_data()

# 2. 특성 선택
selected_features = ['satisfaction_level', 'number_project', 'time_spend_company']
X = df[selected_features]
y = df['left']

# 3. 데이터 분할 및 스케일링
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Streamlit UI 구현
st.title('퇴사율 예측 모델')

satisfaction_level = st.slider('직원 만족도', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
number_project = st.number_input('참여한 프로젝트 수', min_value=1, max_value=10, value=3, step=1)
time_spend_company = st.number_input('회사 근무 기간(년)', min_value=1, max_value=20, value=3, step=1)

if st.button('예측하기'):
    # 6. 사용자 입력 데이터 변환 및 예측
    user_data = np.array([[satisfaction_level, number_project, time_spend_company]])
    user_data_scaled = scaler.transform(user_data)
    prediction = model.predict(user_data_scaled)
    prediction_proba = model.predict_proba(user_data_scaled)[0][1]

    if prediction[0] == 1:
        st.error(f"퇴사 가능성이 높습니다. (확률: {prediction_proba:.2f})")
    else:
        st.success(f"잔류 가능성이 높습니다. (확률: {prediction_proba:.2f})")

# 7. 모델 평가 및 중요 변수 시각화
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("모델 정확도")
st.write(f"Accuracy: {accuracy:.2f}")

# 8. Feature Importance 시각화
feature_importances = pd.DataFrame({
    'Feature': selected_features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

st.subheader("특성 중요도")
fig, ax = plt.subplots()
ax.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')  # 가로 막대 그래프
ax.set_xlabel("중요도")
ax.set_ylabel("특성", rotation=90) 
ax.set_title("특성 중요도")

# 글자 가독성을 높이기 위해 x축 눈금 조정
plt.xticks(rotation=0)  # X축 글자를 가로 방향으로 설정

# Streamlit에 Matplotlib 그래프 표시
st.pyplot(fig)
