import os
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader,TextLoader
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


load_dotenv()


current_dir=os.path.dirname(os.path.abspath(__file__))
db_dir=os.path.join(current_dir,"db")
per_dir=os.path.join(current_dir,"db","chroma_db_web")



urls=["https://www.kazuhacloset.com/"]

loader=WebBaseLoader(urls)
documents=loader.load()

text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
docs=text_splitter.split_documents(documents)

print(f"No. of Doc chunks--: {len(docs)}")
print(f"sample chunks:{docs[0].page_content}")

embeddings= HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")


if not os.path.exists(per_dir):
    print("per dir not exist. Initializing vector stores...")
    db=Chroma.from_documents(docs,embeddings,persist_directory=per_dir)
    print(f"----Finished creating vector stores {per_dir}")

else:
    print(f"{per_dir} already exists")
    db=Chroma(persist_directory=per_dir,embedding_function=embeddings)


    

    

retriever=db.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3},
)

query= "what is kazuhacloset?"

relevant_doc=retriever.invoke(query)

print("\n--Relevent doc---")
for i,doc in enumerate(relevant_doc,1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source:{doc.metadata.get('source',"Unknown")}\n")
