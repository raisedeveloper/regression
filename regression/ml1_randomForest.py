import pandas as pd
from sklearn.ensemble import RandomForestClassifier#앙상블 성과지표metrix 혼동행렬 confusion_matrix confusion matrix 시각화를 위해 추가
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns # Confusion matrix 시각화를 위해 추가

# 1. 데이터 로드
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# 2. Pandas DataFrame 생성
df = pd.DataFrame(data=X, columns=feature_names)
df['target'] = y
df['target_names'] = [target_names[i] for i in y]   #target 이름 추가

# 3. 데이터프레임 출력
# print("DataFrame Sample:")
# print(df.head())

# 4. 데이터 분할 (훈련 세트 80%, 테스트 세트 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Random Forest 모델 생성
# 하이퍼파라미터 튜닝을 위해 GridSearchCV 사용
param_grid = {
    'n_estimators': [50, 100, 200], # 트리의 개수
    'max_depth' : [4, 6, 8],        # 트리의 최대 깊이
    'min_samples_split' : [2, 4],   # 노드를 분할하기 위한 최소 샘플 수
    'min_samples_leaf' : [1, 2]     # 리프 노드에 있어야 하는 최소 샘플 수
}

# 6. Random Forest 모델 초기화
rf_model = RandomForestClassifier(random_state=42)

# 7. 교차 검증 cv = 3 : 훈련 데이터를 3개의 FOLD(묶음)로 나누어서 두개를 훈련, 나머지는 검증 용도로 활용
# 교차 검증은 최적의 하이퍼 파라미터 값을 찾기 위해서 사용됨
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='accuracy')

# 8. 모델 학습 (GridSearchCV를 통한 최적의 파라미터를 반영한 학습)
grid_search.fit(X_train, y_train)

# 9. 최적의 파라미터 출력
# print("Best Parameters:", grid_search.best_params_)

# 10. 최적의 모델 저장
best_rf_model = grid_search.best_estimator_

# 11. 테스트 데이터로 예측 (하이퍼 파라미터 튜닝을 통해 최적화된 상태로 예측이 이루어짐)
y_pred = best_rf_model.predict(X_test)

# 12. 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

# 13. 분류 보고서
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

