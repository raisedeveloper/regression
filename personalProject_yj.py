import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from joblib import dump, load

# 폰트지정
plt.rcParams['font.family'] = 'Malgun Gothic'

# 마이너스 부호 깨짐 지정
plt.rcParams['axes.unicode_minus'] = False

# 숫자가 지수표현식으로 나올 때 지정
pd.options.display.float_format = '{:.2f}'.format

# 1. 데이터 로드 및 캐싱
@st.cache_data
def load_data():
    df = pd.read_csv('dataset/mental_health_wearable_data.csv', encoding='cp949')
    df = df.rename(columns={'Mental_Health_Condition ': 'Mental_Health_Condition'})
    return df

df = load_data()
df.columns = df.columns.str.strip()
    
# 2. 특성 선택
selected_features = ['Heart_Rate_BPM', 'Sleep_Duration_Hours', 'Physical_Activity_Steps', 'Mood_Rating']
X = df[selected_features]
y = df['Mental_Health_Condition']

# 3. 데이터 분할 및 스케일링
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 최적의 하이퍼파라미터 찾기 (GridSearchCV 실행)
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5]
}
print(param_grid_rf)
rf = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf.fit(X_train_scaled, y_train)

# 5. 최적의 모델 학습
model = RandomForestClassifier(
    n_estimators=grid_search_rf.best_params_['n_estimators'],
    max_depth=grid_search_rf.best_params_['max_depth'],
    min_samples_split=grid_search_rf.best_params_['min_samples_split'],
    random_state=42
)
model.fit(X_train_scaled, y_train)

# 6. 모델 저장
dump(model, 'model_model.joblib')

# 7. Streamlit UI
st.title('🧘‍♀️정신 건강 예측 모델🧘‍♂️')

heart_rate = st.slider('심박수 (BPM)', min_value=40, max_value=200, value=80, step=1)
sleep_duration = st.slider('수면 시간 (시간)', min_value=0.0, max_value=12.0, value=7.0, step=0.1)
physical_steps = st.number_input('일일 걸음 수', min_value=0, max_value=30000, value=5000, step=100)
mood_rating = st.slider('기분 평가 (1~10)', min_value=1, max_value=10, value=5, step=1)

# 8. 결과 예측
if st.button('예측하기'): # 사용자 입력 데이터 변환 및 예측
    user_data = np.array([[heart_rate, sleep_duration, physical_steps, mood_rating]])
    user_data_scaled = scaler.transform(user_data)
    prediction = model.predict(user_data_scaled)
    prediction_proba = model.predict_proba(user_data_scaled)[0][1]
    
    if prediction[0] == 1:
        st.error(f'정신 건강 문제 위험이 높습니다. (확률: {prediction_proba:.2f})')
    else:
        st.success(f'정신 건강 문제 위험이 낮습니다. (확률: {prediction_proba:.2f})')

# 9. 모델 평가
st.subheader("✅모델 정확도")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy:.2f}")

# 10. 특성 중요도 시각화
feature_importances = pd.DataFrame({
    'Feature': selected_features, 
    'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

st.subheader('특성 중요도')
fig, ax = plt.subplots()
ax.barh(feature_importances['Feature'], feature_importances['Importance'], color='pink') # 가로 막대그래프
ax.set_xlabel('중요도')
ax.set_ylabel('특성', rotation=90)
ax.set_title('특성 중요도')
st.pyplot(fig)

# 11. 혼동 행렬, 히트맵 (Confusion Matrix) 시각화
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=selected_features, yticklabels=y)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# # 12. 상관관계 히트맵 - Streamlit에 Matplotlib 그래프 표시
# st.subheader('상관관계 히트맵')
# fig, ax = plt.subplots(figsize=(8, 6))
# sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
# st.pyplot(fig)