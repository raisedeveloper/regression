import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load

# 1. 데이터 로드
file_path = "dataset/mental_health_wearable_data.csv"
df = pd.read_csv(file_path)

# 2. Null값 확인
print("결측치 개수:\n", df.isnull().sum())

# 3. 공백 제거 (문자열 데이터가 있다면 공백 제거)
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# 4. 인코딩 (이진 분류이므로 LabelEncoder 사용 가능)
label_encoder = LabelEncoder()
df['Mental_Health_Condition'] = label_encoder.fit_transform(df['Mental_Health_Condition'])

# 5. 특성 선택
selected_features = ['Heart_Rate_BPM', 'Sleep_Duration_Hours', 'Physical_Activity_Steps', 'Mood_Rating']
X = df[selected_features]
y = df['Mental_Health_Condition']

# 6. 데이터 분할 (80% 학습, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. 스케일링 (StandardScaler 적용)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. 최적의 그리드 찾기 (랜덤 포레스트, SVM 비교)
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
rf = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf.fit(X_train_scaled, y_train)

param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}
svm = SVC()
grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=5, scoring='accuracy')
grid_search_svm.fit(X_train_scaled, y_train)

# 9. 캐시에 저장 (최적 모델 저장)
dump(grid_search_rf.best_estimator_, 'best_rf_model.joblib')
dump(grid_search_svm.best_estimator_, 'best_svm_model.joblib')

# 10. 다른 모델과 비교
rf_best = grid_search_rf.best_estimator_
svm_best = grid_search_svm.best_estimator_

rf_pred = rf_best.predict(X_test_scaled)
svm_pred = svm_best.predict(X_test_scaled)

rf_acc = accuracy_score(y_test, rf_pred)
svm_acc = accuracy_score(y_test, svm_pred)

print("Random Forest Accuracy:", rf_acc)
print("SVM Accuracy:", svm_acc)

# 11. 최적의 모델 선택
best_model = rf_best if rf_acc > svm_acc else svm_best
print("최적의 모델:", best_model)

# 12. 딕셔너리로 결과 저장
evaluation_results = {
    "Random Forest Accuracy": rf_acc,
    "SVM Accuracy": svm_acc,
    "Best Model": best_model
}

# 13. plt 시각화
plt.figure(figsize=(10, 5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()
