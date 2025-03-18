# SVM(support vector machine) 초평면과 서포트 벡터 사이의 마진을 최대로 하는 방향으로 최적화 수행

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 1. 데이터 불러오기
data = pd.read_csv('./dataset/spam.csv', encoding='ISO-8859-1')
data = data[['v1','v2']] # 필요한 열만 선택
data.columns = ['label', 'text'] # 열 이름 변경

# 2. 데이터 전처리
data['label'] = data['label'].map({'ham': 0, 'spam': 1})
data.dropna(inplace=True) # 결측치 제거

# 3. 학습 및 테스트 데이터 분리
X_train, X_test, Y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# 4. TF-IDF 벡터화
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. SVM 모델 학습
model = SVC(kernel='linear', random_state=42) #SVM 모델 생성 (linear : 직선, 평면, 초평면)
model.fit(X_train_tfidf, Y_train)
# 6. 예측 및 평가
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. 샘플 테스트
sample_text = ["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 now!",
    "Congratulations! You've been selected to receive a free iPhone 15. Click to claim now!",
    "Hey, are we meeting for lunch tomorrow?"
]
sample_tfidf = vectorizer.transform(sample_text)
predictions = model.predict(sample_tfidf)
for text, label in zip(sample_text, predictions):
    print(f"Text: {text} => {'Spam' if label == 1 else 'Ham'}")
