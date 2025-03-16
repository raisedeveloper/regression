import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# 1. 데이터 준비
iris = load_iris()
X = iris.data  # 특징 (꽃잎, 꽃받침 길이/너비)
y = iris.target  # 품종 (0: Setosa, 1: Versicolor, 2: Virginica)
feature_names = iris.feature_names # 특징 이름 가져오기
target_names = iris.target_names # target 이름 가져오기

# Pandas DataFrame 생성 및 출력
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y  # 품종 레이블 추가
df['target_name'] = df['target'].apply(lambda x: target_names[x])  # 품종 이름 추가

print("Iris Data (Head 5):")
print(df.head())
print("\n")

# 2. 훈련 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=42)
# test_size=0.3: 테스트 데이터 30%, 훈련 데이터 70%로 분리
# random_state=42: 재현성을 위해 난수 시드 고정

# 3. KNN 모델 생성 및 학습
k = 5  # K 값 설정 (이웃의 수)
knn = KNeighborsClassifier(n_neighbors=k) # 모델 생성
knn.fit(X_train, y_train) # 훈련 데이터로 모델 학습

# 4. 테스트 데이터로 예측
y_pred = knn.predict(X_test) # 테스트 데이터에 대한 예측 수행

# 5. 모델 평가
accuracy = accuracy_score(y_test, y_pred) # 정확도 계산
print(f"Accuracy: {accuracy:.4f}") # 결과 출력


# 6. 새로운 데이터에 대한 얘축 (선택 사항)
new_data = np.array([[5.1, 3.5, 1.4, 0.2],[5.9, 3.4, 1.8, 0.3]])  # 새로운 데이터 (꽃잎, 꽃받침 길이/너비)
prediction = knn.predict(new_data)
print(f"New data prediction: {iris.target_names[prediction[0]]}")
print(f"New data prediction: {iris.target_names[prediction[1]]}")