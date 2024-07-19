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
    search_kwargs={"score_threshold": 0.2}
)
mmr_retriever = vectorstore.as_retriever(search_type="mmr")
similarity_score_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.2}
)
# retriever!
retriever = similarity_retriever

# ---------------------------------------------------------------------

# Generate
'''
system_prompt_str = """
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Answer for the question in Korean.

{context} """.strip()

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_str),
        ("human", "{input}"),
    ]
)

azure_model = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
)

question_answer_chain = create_stuff_documents_chain(
    azure_model, prompt_template)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 쿼리를 벡터로 변환하고, 유사한 문서 검색
# results = vectorstore.similarity_search(query)

query_list = [
    # "안녕, 나는 샘이라고 해",
    "왕녀가 가지고 놀던 것은?",
    "개구리의 정체는 뭐야?",
    "둘은 마지막에 어떻게 되지?",
    # "내 이름이 뭐라 그랬지?"
]

for query in query_list:
    print(f"Me : {query}")
    chain_output = rag_chain.invoke({"input": query})
    # print("\n",chain_output,"\n")
    print(f"LLM : {chain_output["answer"]}")
    
'''
# 예제 쿼리
query = "강원도 저렴한 숙소"
docs = retriever.invoke(query)
for doc in docs:
    print("-"*100)
    print(doc.page_content)
