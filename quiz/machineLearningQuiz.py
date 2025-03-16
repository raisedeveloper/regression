import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

dataset = pd.read_csv('./dataset/Advertising.csv')

X = dataset[['TV', 'Radio', 'Newspaper']]
y = dataset['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

intercept = model.intercept_
coefficients = model.coef_

# 1차 함수
equation = f"Sales = {intercept:.4f} + {coefficients[0]:.4f}*TV + {coefficients[1]:.4f}*Radio + {coefficients[2]:.4f}*Newspaper"

# 새로운 광고 예산
new_data = np.array([
    [200, 50, 30],  
    [150, 30, 40],  
    [300, 70, 20]   
])

predictions = model.predict(new_data)
for i, pred in enumerate(predictions):
    print(f"예산 조합 {i+1} (TV={new_data[i][0]}, Radio={new_data[i][1]}, Newspaper={new_data[i][2]}) → 예측 판매량: {pred:.2f}")
    
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.scatter(y_test, y_test_pred, color='blue', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('실제 판매량 vs 예측 판매량 (테스트 데이터)')
plt.xlabel('실제 판매량')
plt.ylabel('예측 판매량')
plt.grid(True, linestyle='--', alpha=0.5)

plt.subplot(2, 2, 2)
plt.scatter(X_train['TV'], y_train, color='green', alpha=0.5, label='훈련 데이터')
plt.scatter(X_test['TV'], y_test, color='blue', alpha=0.5, label='테스트 데이터')

tv_range = np.linspace(X['TV'].min(), X['TV'].max(), 100).reshape(-1, 1)
tv_model = LinearRegression().fit(X_train[['TV']], y_train)
tv_y_pred = tv_model.predict(tv_range)
plt.plot(tv_range, tv_y_pred, color='red', linewidth=2)


plt.title('TV 광고와 판매량의 관계')
plt.xlabel('TV 광고 예산')
plt.ylabel('판매량')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.subplot(2, 2, 3)
plt.scatter(X_train['Radio'], y_train, color='green', alpha=0.5, label='훈련 데이터')
plt.scatter(X_test['Radio'], y_test, color='blue', alpha=0.5, label='테스트 데이터')

radio_range = np.linspace(X['Radio'].min(), X['Radio'].max(), 100).reshape(-1, 1)
radio_model = LinearRegression().fit(X_train[['Radio']], y_train)
radio_y_pred = radio_model.predict(radio_range)
plt.plot(radio_range, radio_y_pred, color='red', linewidth=2)

plt.title('Radio 광고와 판매량의 관계')
plt.xlabel('Radio 광고 예산')
plt.ylabel('판매량')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.subplot(2, 2, 4)
plt.scatter(X_train['Newspaper'], y_train, color='green', alpha=0.5, label='훈련 데이터')
plt.scatter(X_test['Newspaper'], y_test, color='blue', alpha=0.5, label='테스트 데이터')

np_range = np.linspace(X['Newspaper'].min(), X['Newspaper'].max(), 100).reshape(-1, 1)
np_model = LinearRegression().fit(X_train[['Newspaper']], y_train)
np_y_pred = np_model.predict(np_range)
plt.plot(np_range, np_y_pred, color='red', linewidth=2)

plt.title('Newspaper 광고와 판매량의 관계')
plt.xlabel('Newspaper 광고 예산')
plt.ylabel('판매량')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.suptitle('광고 매체별 판매량 영향 분석: 다중 선형 회귀 모델', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()

plt.figure(figsize=(8, 6))
feature_importance = pd.DataFrame({
    '특성': X.columns,
    '계수': np.abs(model.coef_)
})
feature_importance = feature_importance.sort_values('계수', ascending=False)

plt.bar(feature_importance['특성'], feature_importance['계수'], color=['green', 'blue', 'orange'])
plt.title('광고 매체별 영향력 (계수의 절대값)')
plt.xlabel('광고 매체')
plt.ylabel('계수 절대값')
plt.grid(True, linestyle='--', axis='y', alpha=0.5)
plt.show()

plt.figure(figsize=(10, 5))
residuals = y_test - y_test_pred
plt.subplot(1, 2, 1)
plt.scatter(y_test_pred, residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.title('예측값 vs 잔차')
plt.xlabel('예측값')
plt.ylabel('잔차')
plt.grid(True, linestyle='--', alpha=0.5)

plt.subplot(1, 2, 2)
plt.hist(residuals, bins=15, edgecolor='black')
plt.title('잔차 분포')
plt.xlabel('잔차')
plt.ylabel('빈도')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()  # tight_layout : 글자 곂침 방지
plt.show()