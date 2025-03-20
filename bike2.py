import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# 1. 데이터 로드
file_path = "dataset/bike.csv"
df = pd.read_csv(file_path)

# 2. 불필요한 'datetime' 컬럼 제거
df = df.drop(columns=['datetime'])

# 3. 입력 변수(X)와 목표 변수(y) 분리
X = df.drop(columns=['count'])  # 'count'를 제외한 나머지 특성 사용
y = df['count']

# 4. 학습/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 데이터 스케일링 (필요한 경우)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. 여러 회귀 모델 비교
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "Support Vector Regressor": SVR()
}

results = {}

for name, model in models.items():
    # 모델 학습
    if name == "Support Vector Regressor":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # 성능 평가
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = (mse, r2)

# 7. 가장 성능이 좋은 모델 선택
best_model_name = min(results, key=lambda k: results[k][0])  # MSE가 가장 낮은 모델 선택
best_model = models[best_model_name]

# 8. 최적 모델 재학습 및 예측
if best_model_name == "Support Vector Regressor":
    best_model.fit(X_train_scaled, y_train)
    y_pred_best = best_model.predict(X_test_scaled)
else:
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)

# 9. 예측값과 실제값 비교 시각화
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_best, alpha=0.5, color="blue", label="Predicted vs Actual")
plt.plot([0, max(y_test)], [0, max(y_test)], linestyle="--", color="red", label="Perfect Fit")
plt.xlabel("Actual Count")
plt.ylabel("Predicted Count")
plt.title(f"Best Regression Model: {best_model_name}")
plt.legend()
plt.show()

# 10. 최종 모델 평가 출력
print(f"최적 모델: {best_model_name}")
print(f"MSE: {results[best_model_name][0]:.2f}")
print(f"R² Score: {results[best_model_name][1]:.2f}")
