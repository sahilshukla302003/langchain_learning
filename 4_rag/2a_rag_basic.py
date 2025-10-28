import os
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


load_dotenv()


current_dir=os.path.dirname(os.path.abspath(__file__))
book_dir=os.path.join(current_dir,"books")
db_dir=os.path.join(current_dir,"db")
per_dir=os.path.join(current_dir,"db","chroma_db_metadata")


if not os.path.exists(per_dir):
    print("per dir not exist. Initializing vector stores...")

    if not os.path.exists(book_dir):
        raise FileNotFoundError(
            f"the file {book_dir}  does not exist. pls check the path"
        )
    
    book_files=[f for f in os.listdir(book_dir) if f.endswith(".txt")]
    
    documents=[]

    for book_file in book_files:
        file_path=os.path.join(book_dir,book_file)
        loader=TextLoader(file_path)
        book_docs=loader.load()
        for doc in book_docs:
            doc.metadata={"source":book_file}
            documents.append(doc)



    text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    docs=text_splitter.split_documents(documents)

    print("====Doc Chunk Info===")

    print(f"No. of Doc chunks--: {len(docs)}")
    print(f"sample chunks:{docs[0].page_content}")


# YAHA TAK CHUNKS ME TOOTI HAI FILE 

    print("====Creating Embedding======")

    embeddings= HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
    print("====Finished Creating Embedding====")
    

    db=Chroma.from_documents(
        docs,embeddings,persist_directory=per_dir
    )
    print("=====Finished creating vector stores=====")

else:
    print("Vector store already exists, No need to initialize.")

    