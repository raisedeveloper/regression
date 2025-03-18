# 나이브 베이즈(Naive Bayes) : 텍스트 분류 문제에서 매우 효과적

# 스팸분류 예제
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer # 텍스트 데이터를 숫자로 변환
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. 데이터 불러오기
data = pd.read_csv('./dataset/spam.csv', encoding='ISO-8859-1')  # encoding 방식 : utf-8(범용성), ISO-8859-1(서유럽), CP949(한글)

# 데이터 내용 미리보기 (처음 5개 행 출력)
print("Spam Data (First 5 rows):")
print(data.head())
print("\n")

data = data[['v1', 'v2']] # 필요한 열만 선택
data.columns = ['label', 'text'] # 열 이름 변경

# 2. 데이터 전처리
data['label'] = data['label'].map({'ham':0, 'spam': 1}) # 레이블을 숫자로 변환, map함수: 열의 각 값을 다른 값으로 변환하는 데 사용
data.dropna(inplace=True) # 결측치 제거

# 3. 학습 및 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# 4. TF-IDF 벡터화
vectorizer = TfidfVectorizer(stop_words='english')   # stop_words: 불필요한 영어 단어(예: "the", "is", "and" 등)를 자동으로 제거
X_train_tfidf = vectorizer.fit_transform(X_train)   # fit : 학습 데이터의 중요한 통계정보(단어빈도수)학습, transform: 숫자 벡터로 변환
X_test_tfidf = vectorizer.transform(X_test)

# 5. 모델 학습
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 6. 예측 및 평가
y_pred = model.predict(X_train_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassfication Report:\n", classification_report(y_test, y_pred))

# 7. 샘플 테스트
sample_text = ["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 now!",
               "Congratulations! You've been selected to receive a free iPhone 15. Click to claim now!",
               "Hey, are we meeting for lunch tomorrow?"]
sample_tfidf = vectorizer.transform(sample_text)
predictions = model.predict(sample_tfidf)
for text, label in zip(sample_text, predictions):
    print(f"Text: {text} => {'Spam' if label == 1 else 'Ham'}")