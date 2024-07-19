# app.py

import streamlit as st
from langchain_community.vectorstores import Chroma
import pandas as pd
from langchain_core.documents import Document
import os
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

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
        vectorstore = Chroma(persist_directory=vectorstore_path,
                             embedding_function=embedding_model)
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
        vectorstore = Chroma.from_documents(
            documents, embedding_model, persist_directory=vectorstore_path)
        return vectorstore


# Streamlit 애플리케이션
st.title("Document Retrieval System")

# 벡터 저장소 로드
vectorstore = load_vectorstore()

similarity_retriever = vectorstore.as_retriever(search_type="similarity")

# ---
# Generate

system_prompt_str = """
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Show result in html. 

숙소 정보와 질문자가 궁금해 할 만한 정보를 html 형태로 정리해줘.

{context} """.strip()

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_str),
        ("user", "만리포 전경"),
        ("assistant", """ 
<body>
    <h1>아드리아 모텔</h1>
    <p><strong>주소:</strong> 충청남도 태안군 소원면 만리포1길 9</p>
    <p><strong>개요:</strong> 만리포 아드리아모텔은 해안국립공원인 태안군 만리포해수욕장에 위치해 있습니다. 서해안의 대표적인 해수욕장 
인 만리포는 고운 모래와 완만한 경사, 얕은 수심으로 가족단위의 해수욕장으로 유명합니다.</p>
    <p><strong>연락처:</strong> 041-672-6711</p>
    <p><strong>체크인:</strong> 14:00</p>
    <p><strong>체크아웃:</strong> 11:00</p>

    <h2>객실 정보</h2>
    <h3>특실</h3>
    <ul>
        <li><strong>기준 인원:</strong> 2명</li>
        <li><strong>비수기 주중:</strong> 50,000원</li>
        <li><strong>비수기 주말:</strong> 60,000원</li>
        <li><strong>시설:</strong> 목욕시설, 에어컨, TV, 케이블, 인터넷, 냉장고, 테이블, 드라이기</li>
    </ul>

    <h3>일반실</h3>
    <ul>
        <li><strong>기준 인원:</strong> 2명</li>
        <li><strong>비수기 주중:</strong> 40,000원</li>
        <li><strong>비수기 주말:</strong> 50,000원</li>
        <li><strong>시설:</strong> 목욕시설, 에어컨, TV, 케이블, 인터넷, 냉장고, 테이블, 드라이기</li>
    </ul>

    <h2>추가 정보</h2>
    <p>아드리아 모텔은 주차 가능하며 조리가 불가능합니다. 부대 시설로는 세미나실, 스포츠시설, 사우나실, 노래방, 바베큐장, 캠프화이어,  
자전거 대여, 휘트니스 센터, 공용 PC실, 공용 샤워실이 제공되지 않습니다.</p>
</body>
</html>"""),
        ("human", "{input}"),
    ]
)

azure_model = AzureChatOpenAI(
    api_key="2374f2c1a634407387e2fb2fbba5e7fe",  # 여기서 직접 API 키를 전달
    azure_endpoint='https://magicecoleai.openai.azure.com/',
    azure_deployment="gpt-4o",
    api_version="2024-05-01-preview"
)

question_answer_chain = create_stuff_documents_chain(
    azure_model, prompt_template)
rag_chain = create_retrieval_chain(similarity_retriever, question_answer_chain)


# 검색 쿼리 입력
query = st.text_input("Enter your query:")

if query:
    docs = similarity_retriever.invoke(query)
    for doc in docs:
        print("-" * 100)
        print(doc.page_content)

    chain_output = rag_chain.invoke({"input": query})
    print("\n", chain_output, "\n")
    print(f"LLM : {chain_output["answer"]}")

    # st.write(f"Found {len(docs)} documents:")
    # for doc in docs:
    #     for doc in docs:
    #         st.write("-" * 100)
    #         st.write(doc.page_content)
    st.write(f"검색 결과:")
    html = chain_output["answer"]
    # st.markdown(html, unsafe_allow_html=True)
    st.html(chain_output["answer"])
