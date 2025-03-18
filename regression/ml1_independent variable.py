# 필요한 라이브러리 임포트
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
# 윈도우의 경우 주로 'Malgun Gothic'(맑은 고딕), 맥OS의 경우 'AppleGothic'
# 시스템에 설치된 한글 폰트로 대체할 수 있습니다
plt.rcParams['font.family'] = 'Malgun Gothic'  # 또는 다른 한글 폰트
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 1. 데이터 로드
df = pd.read_csv('dataset/Advertising.csv')

# 2. 데이터 전처리
# 'Unnamed: 0' 컬럼 제거
df = df.drop('Unnamed: 0', axis=1)

# 독립 변수(X)와 종속 변수(y) 설정
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# 3. 데이터 분할 (8:2 비율)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 5. 테스트 데이터 예측
y_pred = model.predict(X_test)

# 6. 모델 평가 (R-squared 계산)
r2 = r2_score(y_test, y_pred)

# 7. 새로운 데이터 예측
new_data = pd.DataFrame({'TV': [200], 'Radio': [50], 'Newspaper': [30]})
new_prediction = model.predict(new_data)

# 8. 결과 출력
print(f"R-squared Score: {r2:.4f}")
print(f"새로운 광고 예산(TV=200, Radio=50, Newspaper=30)에 대한 예측 판매량: {new_prediction[0]:.2f}")

# 9. 시각화
# (1) 실제 vs 예측 판매량 산점도
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('실제 판매량')
plt.ylabel('예측 판매량')
plt.title('실제 판매량 vs 예측 판매량')
plt.grid(True)
plt.show()

# (2) 각 독립변수와 판매량의 관계 시각화
plt.figure(figsize=(15, 5))

# TV vs Sales
plt.subplot(1, 3, 1)
sns.scatterplot(x=df['TV'], y=df['Sales'])
sns.regplot(x=df['TV'], y=df['Sales'], scatter=False, color='red')
plt.xlabel('TV 광고 예산')
plt.ylabel('판매량')
plt.title('TV vs Sales')

# Radio vs Sales
plt.subplot(1, 3, 2)
sns.scatterplot(x=df['Radio'], y=df['Sales'])
sns.regplot(x=df['Radio'], y=df['Sales'], scatter=False, color='red')
plt.xlabel('Radio 광고 예산')
plt.ylabel('판매량')
plt.title('Radio vs Sales')

# Newspaper vs Sales
plt.subplot(1, 3, 3)
sns.scatterplot(x=df['Newspaper'], y=df['Sales'])
sns.regplot(x=df['Newspaper'], y=df['Sales'], scatter=False, color='red')
plt.xlabel('Newspaper 광고 예산')
plt.ylabel('판매량')
plt.title('Newspaper vs Sales')

plt.tight_layout()
plt.show()

# 학습된 모델의 계수 출력 (선택적)
print("\n회귀 모델 계수:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")
print(f"절편: {model.intercept_:.4f}")