import os
from dotenv import load_dotenv
from langchain_chroma import Chroma  # ✅ use updated Chroma package
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from operator import itemgetter


# Load environment
load_dotenv()

# --- Setup paths ---
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_metadata")

# --- Setup embeddings and vector store ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Use latest Chroma package
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

# --- Setup Gemini model ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# --- Define the QA prompt ---
qa_system_prompt = (
    "You are a helpful assistant for question-answering tasks. "
    "Use the retrieved context below to answer the question. "
    "If the answer isn't in the context, say 'I don’t know'. "
    "Keep your response concise (max three sentences).\n\n"
    "Context:\n{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# --- Build new RAG chain manually ---
rag_chain = (
    RunnableMap({
        "context": itemgetter("input") | retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
        "input": itemgetter("input"),
        "chat_history": itemgetter("chat_history")  # ✅ pass it through
    })
    | qa_prompt
    | llm
)
# --- Interactive chat loop ---
def continual_chat():
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        print(f"AI: {result.content}")

        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=result.content))


if __name__ == "__main__":
    continual_chat()
