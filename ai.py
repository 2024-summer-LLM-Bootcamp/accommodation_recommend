from langchain_community.vectorstores import Chroma
import openai
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
import pandas as pd
from langchain_core.documents import Document
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


CURR_DIR = os.path.dirname(os.path.realpath(__file__))
print("CURR_DIR: ", CURR_DIR)

# Azure OpenAI Embeddings 사용
embedding_model = AzureOpenAIEmbeddings(
    api_key="2374f2c1a634407387e2fb2fbba5e7fe",  # 여기서 직접 API 키를 전달
    azure_endpoint='https://magicecoleai.openai.azure.com/',
    model='text-embedding-3-small',  # 사용하려는 Azure OpenAI 모델 이름
)

if os.path.exists(os.path.join(CURR_DIR, "chroma_db")):
    print("load from disk...")
    vectorstore = Chroma(persist_directory="./chroma_db",
                         embedding_function=embedding_model)
else:
    print("create vectorstore and save to disk...")
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
        documents.append(
            Document(page_content=content, metadata={"id": index}))
    vectorstore = Chroma.from_documents(
        documents, embedding_model, persist_directory="./chroma_db")


similarity_retriever = vectorstore.as_retriever(search_type="similarity")
similarity_score_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.01}
)
# retriever!
retriever = similarity_retriever

# 예제 쿼리
# query = "만리포 전경이 보이는 숙소"
# docs = retriever.invoke(query)
# print(len(docs))
# for doc in docs:
#     print("-"*100)
#     print(doc.page_content)

# ---------------------------------------------------------------------

# Generate

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
    api_key="2374f2c1a634407387e2fb2fbba5e7fe",  # 여기서 직접 API 키를 전달
    azure_endpoint='https://magicecoleai.openai.azure.com/',
    azure_deployment="gpt-4o", 
    api_version="2024-05-01-preview"
)

question_answer_chain = create_stuff_documents_chain(
    azure_model, prompt_template)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

query_list = [
    "만리포 전경이 보이는 숙소"
]

for query in query_list:
    print(f"Me : {query}")
    chain_output = rag_chain.invoke({"input": query})
    # print("\n",chain_output,"\n")
    print(f"LLM : {chain_output["answer"]}")
