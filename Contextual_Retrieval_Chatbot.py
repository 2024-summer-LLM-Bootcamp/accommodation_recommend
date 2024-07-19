from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

#여기에 텍스트를 입력하시오.
room_txt="""0,라비치호텔,전북특별자치도 군산시 비응남로 36,"라비치호텔은 비응항에서 바다전경이 가장 좋은 호텔식 숙박업소이다. 세상에서 가장 긴 방조제 새만금, 새로운 서해안 시대를 예고하는 이곳에 대규모 관광타운이 조성되고 있다. 새만금방조제의 북쪽 관문인 비응항 중심에 위치하고 있으며, 유람선 선착장과 항구가 바로 앞에 있다. 아침에는 방조제 위로 떠오르는 일출이 있고, 저녁에는 수평선 너머로 지는 해넘이를 볼 수 있어 인기가 높다. 시설에서도 최상을 자랑한다. 객실마다 커플용 PC 및 월풀욕조등 최고의 서비스를 제공하고 있다. 모든 객실이 바다를 향해 나 있다는 것도 장점 중 하나다. 그래서 일출을 볼 수 있는 방과 일몰을 볼 수 있는 방을 선택할 수도 있다. 주변 먹거리로는 신선한 해산물을 즐기실 수 있는 회센터 및 어시장, 음식점들이 바로 있어 보다 저렴하고 맛있게 바다향을 한껏 담아갈 수 있다.",0.0,,7층,약 100명,34실,"한실, 양실",주차가능,불가,18:00,12:00,0507-1366-2701,,,,없음,없음,없음,없음,없음,없음,없음,없음,없음,없음,없음,"객실명:온돌
객실크기:8평
객실수:0
기준인원:4최대인원:4
※ 위 정보는 2022년 10월에 작성된 정보로(정상요금), 해당 숙박시설 이용요금이 수시로 변동됨에 따라 이용요금 및 기타 자세한 사항은 홈페이지 참조 요망 
비수기주중최소:70000(성수기: 0)
비수기주말최소:70000(성수기: 0)
목욕시설:Y
욕조:Y
홈시어터:에어컨:Y
TV:Y
PC:Y
케이블설치:Y
인터넷:냉장고:Y
세면도구:Y
소파:취사용품:테이블:Y
드라이기:Y
객실명:스위트
객실크기:10평
객실수:0
기준인원:2최대인원:4
※ 위 정보는 2022년 10월에 작성된 정보로(정상요금), 해당 숙박시설 이용요금이 수시로 변동됨에 따라 이용요금 및 기타 자세한 사항은 홈페이지 참조 요망 
비수기주중최소:80000(성수기: 0)
비수기주말최소:80000(성수기: 0)
목욕시설:Y
욕조:Y
홈시어터:에어컨:Y
TV:Y
PC:Y
케이블설치:Y
인터넷:냉장고:Y
세면도구:Y
소파:취사용품:테이블:Y
드라이기:Y
객실명:일반실
객실크기:8평
객실수:0
기준인원:2최대인원:2
※ 위 정보는 2022년 10월에 작성된 정보로(정상요금), 해당 숙박시설 이용요금이 수시로 변동됨에 따라 이용요금 및 기타 자세한 사항은 홈페이지 참조 요망
비수기주중최소:50000(성수기: 0)
비수기주말최소:50000(성수기: 0)
목욕시설:Y
욕조:Y
홈시어터:에어컨:Y
TV:Y
PC:Y
케이블설치:Y
인터넷:냉장고:Y
세면도구:Y
소파:취사용품:테이블:Y
드라이기:Y
객실명:준특실
객실크기:8평
객실수:0
기준인원:2최대인원:2
※ 위 정보는 2021년 12월에 작성된 정보로(정상요금), 해당 숙박시설 이용요금이 수시로 변동됨에 따라 이용요금 및 기타 자세한 사항은 홈페이지 참조 요망 
비수기주중최소:60000(성수기: 0)
비수기주말최소:60000(성수기: 0)
목욕시설:Y
욕조:Y
홈시어터:에어컨:Y
TV:Y
PC:Y
"""


room_document = Document(
    page_content=room_txt,
)


recursive_text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n",".",","],
    chunk_size=200,
    chunk_overlap=20,
    length_function=len,
)

recursive_splitted_document = recursive_text_splitter.split_documents([room_document])


embedding_model=AzureOpenAIEmbeddings(
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


#---------------------------------------------------------------------

#Generate

system_prompt_str = """
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Answer the question in the same language as the question. If you don't know the answer, simply say 'I don't know' for English questions or '알 수 없습니다' for Korean questions.

{context} """.strip()

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_str),
        ("human", "{input}"),
    ]
)

azure_model = AzureChatOpenAI(
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT"),
)

question_answer_chain = create_stuff_documents_chain(azure_model, prompt_template)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

messages = [
    SystemMessage(content="You are a helpful assistant. Answer all questions to the best of your ability."),
    ]

print("Assistant: Hello! How can I assist you today? (To exit, type 'bye')")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Assistant: Goodbye!")
        break

    messages.append(HumanMessage(content=user_input))

    chain_output = rag_chain.invoke({"input": user_input})

    print(f"Assistant: {chain_output['answer']}")

    messages.append(AIMessage(content=chain_output['answer']))
