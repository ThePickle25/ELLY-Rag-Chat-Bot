import os
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from seed_data import seed_from_json, seed_from_pdf, connect_to_milvus
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain.agents import AgentExecutor, create_tool_calling_agent
from dotenv import load_dotenv

load_dotenv()


def get_retriever(collection_name: str, URL_link: str = None, ) -> EnsembleRetriever:
    if URL_link is None:
        URL_link = 'http://localhost:19530'
    try:
        vectorstore = connect_to_milvus(URL_link, collection_name)
        milvus_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 15}
        )
        document = [
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in vectorstore.similarity_search("", k=100)
        ]

        if not document:
            raise ValueError(f"Can not found document in collection {collection_name}")
        bm52_retriever = BM25Retriever.from_documents(document)
        bm52_retriever.k = 15
        ensemble_retriever = EnsembleRetriever(
            retrievers=[milvus_retriever, bm52_retriever],
            weight=[0.7, 0.3]
        )
        return ensemble_retriever
    except Exception as e:
        print(f"Error {e}")
        error_doc = [
            Document(
                page_content="Fail to connect with database!",
                metadata={"page": "error"}
            )
        ]
        return BM25Retriever.from_documents(error_doc)


def get_agent(retriever):
    tool = create_retriever_tool(
        retriever,
        "find_documents",
        "Use this tool to retrieve relevant internal documents before answering any question. Always use this first."

    )

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash",
        api_key="AIzaSyBjYWfBi3Yo2dT46Maw6sc0hzIxyoaicn8"
    )

    tools = [tool]

    system = system = """
Your name is Elly, an AI assistant connected to an internal document retrieval system via a tool called `find_documents`.

You MUST use the `find_documents` tool if the user's question contains or implies a request to check internal documents.

Some examples of such phrases include:
- "based on the uploaded document"
- "according to the document I provided"
- "in the dataset"
- "does the document mention..."
- "check the uploaded PDF"
- "search the internal document for..."
- "look in the documents"
- "within the provided materials"

The user has already uploaded internal documents (e.g., from PDF or other sources), and these documents are indexed and ready to search using the `find_documents` tool. You do not need to ask the user to upload files again.

When a user says something like:
- "check the uploaded PDF"
- "find in the document"
- "based on the file I provided"
Assume they are referring to the internal documents already indexed. You MUST use the `find_documents` tool to find relevant answers.

Also, if the question has a similar meaning — even without exact phrases — you should assume the user wants information retrieved from internal documents and therefore MUST call the `find_documents` tool before answering.

"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ])
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


if __name__ == "__main__":
    # Tên collection và link Milvus
    collection_name = "data_test"
    milvus_url = "http://localhost:19530"  # hoặc URL Milvus cloud nếu có
    # vectorstore = connect_to_milvus(milvus_url,collection_name)
    # docs = vectorstore.similarity_search("lane detection", k=5)
    # print("Result:", docs)

    # Khởi tạo retriever
    retriever = get_retriever(collection_name=collection_name, URL_link=milvus_url)

    # Tạo RAG Agent
    agent_executor = get_agent(retriever)

    # Giao diện dòng lệnh đơn giản
    print("🔍 RAG Chatbot is ready! Type your question (type 'exit' to quit)\n")

    chat_history = []  # Giữ lịch sử hội thoại nếu cần
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break

        response = agent_executor.invoke({
            'input': query,
            'chat_history': chat_history
        })

        print("Assistant:", response["output"])
        # Ghi vào lịch sử nếu muốn sử dụng trong prompt
        chat_history.append(("human", query))
        chat_history.append(("ai", response["output"]))
