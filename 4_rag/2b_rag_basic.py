from langchain_chroma import Chroma
import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

current_dir=os.path.dirname(os.path.abspath(__file__))
per_dir=os.path.join(current_dir,"db","chroma_db_metadata")



embeddings= HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

db=Chroma(
        persist_directory=per_dir,
        embedding_function=embeddings
    )

query="who was Odyssey?"

retriever=db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k":1,"score_threshold":0.1},
)

relevent_docs=retriever.invoke(query)


#display rhe relevant result with metadata

print("\n ------Relevant Doc --------")
for i,doc in enumerate(relevent_docs,1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source:{doc.metadata.get('source',"Unknown")}\n")
