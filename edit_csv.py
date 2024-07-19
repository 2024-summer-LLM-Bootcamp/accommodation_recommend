import pandas as pd

# CSV 파일 경로
filename = 'combined_data.csv'

# CSV 파일 읽기
df = pd.read_csv(filename)

# 삭제할 컬럼 리스트
columns_to_remove = [
    '우편번호',
    '관리자',
    '위도',
    '경도',
    '개요',
    '문의 및 안내',
    '규모',
    '상세정보',
    '환불규정',
    '조리 가능',
    '예약 안내',
    '예약안내 홈페이지',
    '세미나',
    '스포츠시설',
    '사우나실',
    '뷰티 시설',
    '자전거대여',
    '공용 PC실'
    ]

# 컬럼 삭제
df.drop(columns=columns_to_remove, inplace=True)