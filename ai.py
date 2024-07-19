from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# 임베딩 모델 초기화
embeddings = OpenAIEmbeddings()

# 벡터 데이터베이스 초기화
chroma = Chroma()

# 데이터 벡터화 및 인덱싱
for idx, row in accommodations.iterrows():
    vector = embeddings.embed_text(row['description'])
    metadata = {
        'name': row['name'],
        'location': row['location'],
        'price': row['price'],
        'facilities': row['facilities']
    }
    chroma.add_vector(vector, metadata)


def search_accommodations(query):
    # 검색어 벡터화
    query_vector = embeddings.embed_text(query)

    # 벡터 데이터베이스에서 검색
    results = chroma.search(query_vector)

    # 검색 결과 처리
    accommodations = []
    for result in results:
        metadata = result['metadata']
        accommodations.append({
            'name': metadata['name'],
            'location': metadata['location'],
            'price': metadata['price'],
            'facilities': metadata['facilities']
        })

    return accommodations


# 예시 검색
query = "I am looking for a cheap hotel in Seoul with free Wi-Fi."
results = search_accommodations(query)
print(results)
