# 랜덤포레스트 모델 활용한 예제(cf. 다른 회귀모델 성능 비교)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 데이터 로드
data = pd.read_csv('./dataset/HousingData.csv')

# 데이터 전처리
data = data.dropna() # 결측값 제거
X = data[['CRIM', 'ZN', 'INDUS', 'RM, 'AGE, 'DIS', 'TAX', "LSTAT"]]
y = data['MEDV']

# 데이터 분할
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. 랜덤 포레스트 모델
rf_model = RandomForestRegressor(random_state=43, n_estimators=100) # n_estimators: 모델을 구성하는 결정 틀의 개수
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)

# 랜덤 포레스트 평가
rf_mse = mean_squared_error(y_test, rf_y_pred)
rf_r2 = r2_score(y_test, rf_y_pred)
print(f"랜덤 포레스트 - MSE: {rf_mse:.2f}, R2: {rf_r2:.2f}")

# 2. 다른 모델 비교
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42)
}

results ={}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"<SE": mse, "R2": r2}
    print(f"{name} - MSE: {mse:.2f}, R2: {r2:.2f}")
    
# 3. 최적 모델 선택 및 시각화
results["Random Forest"] = {"MSE": rf_mse, "R2": rf_r2}
best_model = max(results, key=lambda x: results[x]['R2'])
print("\n최적 모델:", best_model)

# 결과 시각화
r2_score = [[result = "R2"] for result in results.values()]
model_names = list(results.keys())

plt.figure(figsize=(10, 6))
plt.bar(model_names, r2_scores, color='skyblue')
plt.xlabel("모델")
plt.ylabel("R2 Score")
plt.title("모델별 R2 Score 비교")
plt.grid(axis='y')
plt.show()