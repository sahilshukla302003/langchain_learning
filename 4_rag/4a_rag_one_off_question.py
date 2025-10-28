from langchain_chroma import Chroma
import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage


load_dotenv()

current_dir=os.path.dirname(os.path.abspath(__file__))
per_dir=os.path.join(current_dir,"db","chroma_db_metadata")



embeddings= HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

db=Chroma(
        persist_directory=per_dir,
        embedding_function=embeddings
    )

query="who is Odysseus wife?"

retriever=db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k":10,"score_threshold":0.1},
)

relevent_docs=retriever.invoke(query)


#display rhe relevant result with metadata

print("\n ------Relevant Doc --------")
for i,doc in enumerate(relevent_docs,1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source:{doc.metadata.get('source',"Unknown")}\n")



combined_input=(
    "here are some documents that might help answer the question:"
    +query
    +"\n\n".join([doc.page_content for doc in relevent_docs])
    +"\n\nPlease provide an anser based only on the provided document. if the anser is not from given doc give it from your side" 

)

llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

messages=[
    SystemMessage(content="you are  a helpful assisstent"),
    HumanMessage(content=combined_input)
]

res=llm.invoke(messages)

print(res.content)