from langchain_community.vectorstores import Chroma
import openai
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
import pandas as pd
from langchain_core.documents import Document
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain


CURR_DIR = os.path.dirname(os.path.realpath(__file__))
print("CURR_DIR: ", CURR_DIR)

# CSV 파일 읽기
data = pd.read_csv('accommodation.csv')
cols = data.columns


def create_text_from_row(cols, row):
    content = ""
    for col in cols:
        content += f"{col}: {row[col]}"
    return content


# 각 행을 LangChain의 문서 형식으로 변환
documents = []
for index, row in data.iterrows():
    content = create_text_from_row(cols, row)
    documents.append(Document(page_content=content, metadata={"id": index}))


# OpenAI API 설정
# openai.api_key = '2374f2c1a634407387e2fb2fbba5e7fe'  # 여기에 Azure OpenAI API 키를 직접 입력
# openai.api_base = 'https://magicecoleai.openai.azure.com/'  # 여기에 Azure OpenAI 엔드포인트 URL을 직접 입력

# Azure OpenAI Embeddings 사용
embedding_model = AzureOpenAIEmbeddings(
    api_key="2374f2c1a634407387e2fb2fbba5e7fe",  # 여기서 직접 API 키를 전달
    azure_endpoint='https://magicecoleai.openai.azure.com/',
    model='text-embedding-3-small',  # 사용하려는 Azure OpenAI 모델 이름
)

doc_content_list = [doc.page_content for doc in documents]
embeddings = embedding_model.embed_documents(doc_content_list)

print("\n", f"Number of Embed list of texts : {len(embeddings)}", "\n")
print("\n", f"Sample Vector : {embeddings[0][:5]}")
print("\n", f"Length of Sample Vector {len(embeddings[0])}", "\n")


# Chroma 데이터베이스 생성
vectorstore = Chroma.from_documents(documents, embedding_model)

# 벡터를 디스크에 저장
# vectorstore.save_local(CURR_DIR)

# Chroma 데이터베이스 불러오기
# vectorstore = Chroma.load_local(CURR_DIR, embedding_model)

similarity_retriever = vectorstore.as_retriever(search_type="similarity")
similarity_score_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.01}
)
mmr_retriever = vectorstore.as_retriever(search_type="mmr")
similarity_score_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    # search_kwargs={"score_threshold": 0.0}
)
# retriever!
retriever = similarity_retriever

# 예제 쿼리
query = "만리포 전경이 보이는 숙소"
docs = retriever.invoke(query)
print(len(docs))
for doc in docs:
    print("-"*100)
    print(doc.page_content)
