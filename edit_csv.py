import pandas as pd

# CSV 파일 경로
filename = 'combined_data.csv'

# CSV 파일 읽기
df = pd.read_csv(filename)

# 삭제할 컬럼 리스트
columns_to_remove = [
    '우편번호',
    '전화번호',
    '관리자',
    '위도',
    '경도',
    '예약안내 홈페이지',

    # '개요',
    # '문의 및 안내',
    # '규모',
    # '상세정보',
    # '환불규정',
    # '조리 가능',
    # '예약 안내',
    # '세미나',
    # '스포츠시설',
    # '사우나실',
    # '뷰티 시설',
    # '자전거대여',
    # '공용 PC실'
]

# 컬럼 삭제
df.drop(columns=columns_to_remove, inplace=True)

# 남은 컬럼 리스트
# ['명칭', '주소', '개요', '숙박 종류', '문의 및 안내', '규모', '수용 가능 인원', '객실 수', '객실 유형',
#        '주차 가능', '조리 가능', '체크인', '체크아웃', '예약 안내', '픽업서비스', '식음료장', '부대 시설',
#        '세미나', '스포츠시설', '사우나실', '뷰티 시설', '노래방', '바베큐장', '캠프화이어', '자전거대여',
#        '휘트니스센터', '공용 PC실', '공용 샤워실', '상세정보', '환불규정']

print(df.columns)
df.to_csv("accommodation.csv")
