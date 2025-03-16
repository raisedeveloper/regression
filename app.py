# 사전 설치 : pip install flask pymysql
from flask import Flask, render_template, request, redirect, url_for
from db import Database
import atexit

app = Flask(__name__)
db = Database()

# 애플리케이션 종료 시 DB 연결 종료 
atexit.register(db.close)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')  # GET 요청 처리

@app.route('/result', methods=['POST'])
def calculate():
    try:
        gender = str(request.form['gender'])
        age = str(request.form['age'])
        purchase = str(request.form['purchase'])
        db.save_starbucks_record(gender, age, purchase)
        
        return render_template('result.html',
                                gender=gender,
                                age=age,
                                purchase=purchase)
    except ValueError:
        return render_template('index.html', error="유효한 숫자를 입력해주세요.")
    
@app.route('/history')
def history():
    # 최근 BMI 기록 10개 가져오기
    records = db.get_starbucks_records(10)
    return render_template('history.html', records=records)

if __name__ == '__main__':
    app.run(debug=True)
    