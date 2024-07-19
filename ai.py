from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os

load_dotenv()

with open("/root/LLM_Bootcamp/LangChain_Class/Retrieval/frog_prince.txt", encoding='utf-8') as f:
    frog_txt = f.read()


frog_document = Document(
    page_content=frog_txt,
    metadata={"source": "개구리 왕자 - 방정환 역"}
)


recursive_text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", ".", ","],
    chunk_size=200,
    chunk_overlap=20,
    length_function=len,
)

recursive_splitted_document = recursive_text_splitter.split_documents([
                                                                      frog_document])


embedding_model = AzureOpenAIEmbeddings(
    model="text-embedding-3-small"
)


chroma = Chroma("vector_store")
vector_store = chroma.from_documents(
    documents=recursive_splitted_document,
    embedding=embedding_model
)


similarity_retriever = vector_store.as_retriever(search_type="similarity")
mmr_retriever = vector_store.as_retriever(search_type="mmr")
similarity_score_retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.2}
)

retriever = similarity_retriever


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
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
)

question_answer_chain = create_stuff_documents_chain(
    azure_model, prompt_template)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

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
