o
    ç�g�	  �                   @   s&   d dl Z d dl mZ G dd� d�ZdS )�    N)�Errorc                   @   s.   e Zd Zdd� Zdd� Zddd�Zdd	� Zd
S )�Databasec              
   C   sf   d | _ ztjdddddtjjd�| _ td� W d S  ty2 } ztd|� �� W Y d }~d S d }~ww )	N�	localhost�test�rootzdhforkwk96$Zutf8mb4)�hostZdatabase�user�password�charsetZcursorclassu1   MariaDB에 성공적으로 연결되었습니다.u"   MariaDB 연결 중 오류 발생: )�
connection�pymysql�connectZcursorsZ
DictCursor�printr   )�self�e� r   �!C:\aiproject\classExample01\db.py�__init__   s   ���zDatabase.__init__c              
   C   s�   z9| j du rtd� W dS | j �� �}d}|�|||||f� W d  � n1 s)w   Y  | j ��  td� W dS  tyS } ztd|� �� W Y d}~dS d}~ww )u*   BMI 기록을 데이터베이스에 저장N�*   데이터베이스 연결이 없습니다.Fz�
                INSERT INTO bmi_records (weight, height, bmi, category)
                VALUES (%s, %s, %s, %s)
                u4   BMI 기록이 성공적으로 저장되었습니다.Tu$   데이터 저장 중 오류 발생: )r   r   �cursor�executeZcommitr   )r   �weight�height�bmi�categoryr   �queryr   r   r   r   �save_bmi_record   s    
�
��zDatabase.save_bmi_record�
   c              
   C   s�   z3| j du rtd� g W S | j �� �}d}|�||f� |�� }W d  � |W S 1 s,w   Y  |W S  tyN } ztd|� �� g W  Y d}~S d}~ww )u$   최근 BMI 기록을 가져옵니다Nr   z}
                SELECT * FROM bmi_records
                ORDER BY created_at DESC
                LIMIT %s
                u$   데이터 조회 중 오류 발생: )r   r   r   r   Zfetchallr   )r   �limitr   r   �recordsr   r   r   r   �get_bmi_records)   s"   


�	�	��zDatabase.get_bmi_recordsc                 C   s    | j r| j ��  td� dS dS )u    데이터베이스 연결 종료u(   MariaDB 연결이 종료되었습니다.N)r   �closer   )r   r   r   r   r!   >   s   
�zDatabase.closeN)r   )�__name__�
__module__�__qualname__r   r   r    r!   r   r   r   r   r      s
    
r   )r   r   r   r   r   r   r   �<module>   s    