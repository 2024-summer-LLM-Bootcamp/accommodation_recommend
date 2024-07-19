# app.py

import streamlit as st
from langchain_community.vectorstores import Chroma
import pandas as pd
from langchain_core.documents import Document
import os
from langchain_openai import AzureOpenAIEmbeddings

# 현재 디렉토리 경로
CURR_DIR = os.path.dirname(os.path.realpath(__file__))

# Azure OpenAI Embeddings 설정
embedding_model = AzureOpenAIEmbeddings(
    api_key="2374f2c1a634407387e2fb2fbba5e7fe",  # 직접 API 키를 전달하는 것은 보안상 좋지 않음
    azure_endpoint='https://magicecoleai.openai.azure.com/',
    model='text-embedding-3-small',
)

# 벡터 저장소 로드 또는 생성
vectorstore_path = os.path.join(CURR_DIR, "chroma_db")

def load_vectorstore():
    if os.path.exists(vectorstore_path):
        vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embedding_model)
        return vectorstore
    else:
        # CSV 파일 읽기
        data = pd.read_csv('accommodation.csv')
        cols = data.columns

        def create_text_from_row(cols, row):
            content = ""
            for col in cols:
                content += f"{col}: {row[col]}\n"
            return content

        # 각 행을 LangChain의 문서 형식으로 변환
        documents = []
        for index, row in data.iterrows():
            content = create_text_from_row(cols, row)
            documents.append(
                Document(page_content=content, metadata={"id": index})
            )
        vectorstore = Chroma.from_documents(documents, embedding_model, persist_directory=vectorstore_path)
        return vectorstore

# Streamlit 애플리케이션
st.title("Document Retrieval System")

# 벡터 저장소 로드
vectorstore = load_vectorstore()

# 검색 쿼리 입력
query = st.text_input("Enter your query:")

if query:
    similarity_retriever = vectorstore.as_retriever(search_type="similarity")
    docs = similarity_retriever.invoke(query)
    
    st.write(f"Found {len(docs)} documents:")
    for doc in docs:
        st.write("-" * 100)
        st.write(doc.page_content)
